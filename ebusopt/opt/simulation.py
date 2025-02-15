import copy
import numpy as np
import pandas as pd
import datetime
import logging
import time
from bisect import insort
from scipy.stats import t as tstat
from ebusopt.gtfs_beb.data import get_dh_dict

logger = logging.getLogger('simulation')


# TODO: fill in comments/docstrings
class Vehicle:
    # Class representing individual buses in discrete-event sim
    def __init__(self, id, min_chg, max_chg):
        self.id = id
        self.min_chg = min_chg
        self.max_chg = max_chg
        self.chg = max_chg


class ChargingStation:
    # Class representing charging stations for discrete-event sim
    def __init__(self, id, power, num_chargers):
        self.id = id
        self.power = power
        self.num_chargers = num_chargers
        # List of ChargingRequests currently charging
        self.reqs_charging = list()
        # List of ChargeRequests in queue
        self.queue = list()

    def is_full(self):
        return len(self.reqs_charging) >= self.num_chargers

    def start_charging(self, req):
        if req in self.reqs_charging:
            raise ValueError('Given vehicle is already charging.')
        self.reqs_charging.append(req)

    def add_to_queue(self, req):
        """
        Add a ChargeRequest to the queue.
        :param req: ChargeRequest instance
        """
        self.queue.append(req)

    def finish_charging(self, v_id):
        """
        Remove a vehicle from the currently charging list
        :param v_id: Vehicle ID
        :return:
        """
        req_to_finish = get_object_by_id(v_id, self.reqs_charging)
        self.reqs_charging.remove(req_to_finish)

    def advance_queue(self):
        """
        Progress through queue if one exists.
        :return: request made by vehicle now charging
        """
        if len(self.queue) > 0:
            now_charging_req = self.queue.pop(0)
            self.start_charging(now_charging_req)
            return now_charging_req

        else:
            return None

    def get_queue_time(self):
        """
        Return how long a vehicle would have to wait to start charging
        if it entered the queue at the current time.
        :return:
        """
        chg_times_left = [r.chg_time for r in self.reqs_charging]
        pseudo_q = copy.copy(self.queue)
        q_time = 0

        while len(chg_times_left) >= self.num_chargers:
            min_time = min(chg_times_left)
            q_time += min_time
            chg_times_left.remove(min_time)
            if pseudo_q:
                chg_times_left.append(pseudo_q.pop(0).chg_time)

        return q_time


class ChargeRequest:
    # Class representing a single vehicle's request to use a particular
    # charger at a particular time in discrete event sim.
    def __init__(self, time_made, veh, trip, chg_site, chg_time):
        # We can't have negative charging time.
        if chg_time < 0:
            raise ValueError('Charging time must be nonnegative, {:.5f}'
                             ' is not an acceptable value.'.format(chg_time))
        self.time_made = time_made
        self.veh = veh
        # ID is based only on vehicle
        self.id = self.veh.id
        self.trip = trip
        self.chg_site = chg_site
        self.chg_time = chg_time


class Event:
    # Parent class for all simulation event types
    def __init__(self, time, typ, veh, trip):
        """
        Constructor for simulation event class
        """
        # Datetime at which event occurs
        self.time = time
        # Type of event, e.g. 'trip_end'
        self.type = typ
        # Vehicle involved
        self.veh = veh
        # Trip index
        self.trip = trip

    def __lt__(self, other):
        """
        Less-than comparison operator, used for sorting events
        """
        return self.time < other.time


class TripStart(Event):
    def __init__(self, time, veh, trip):
        super().__init__(time, 'trip_start', veh, trip)


class TripEnd(Event):
    def __init__(self, time, veh, trip):
        super().__init__(time, 'trip_end', veh, trip)


class ChargerArrival(Event):
    def __init__(self, time, veh, trip, chg_site, chg_time):
        super().__init__(time, 'chg_arr', veh, trip)
        # Location of charger
        self.chg_site = chg_site
        # Scheduled charging time for vehicle
        self.chg_time = chg_time


class ChargerDeparture(Event):
    def __init__(self, time, veh, trip, chg_site, chg_time):
        super().__init__(time, 'chg_dpt', veh, trip)
        self.chg_site = chg_site
        self.chg_time = chg_time


class Calendar:
    def __init__(self):
        """
        Constructor for simulation calendar class
        """
        self.events = list()

    def __len__(self):
        return len(self.events)

    def add_event(self, event):
        if not isinstance(event, Event):
            raise TypeError('Only objects of type Event may be added'
                            ' to calendar.')

        # insort() right-inserts in the event of a tie, so our queue
        # will reflect the order in which items were added, as
        # intended.
        insort(self.events, event)

    def remove_event(self):
        """
        Remove first event from calendar. Call after event has been
        processed.
        """
        _ = self.events.pop(0)

    def get_next_event(self):
        return self.events[0]

    def head(self, i=5):
        i_adj = max(i, len(self.events))
        return self.events[:i_adj]


def get_object_by_id(obj_id, obj_list):
    """
    Return the object that corresponds to the given ID. Can be applied
    to Vehicle or ChargingStation (or generally, any class with .id
    field.

    :param obj_id: string giving charging object ID
    :param obj_list: list of matching objects, one of which has field
        id that matches site_id
    :return: object instance from given list
    """
    obj_matches = [o for o in obj_list if o.id == obj_id]

    if len(obj_matches) == 0:
        raise ValueError('No match for ID found in given list.')

    elif len(obj_matches) > 1:
        raise ValueError('Found {} objects in list instead of 1'
                         'with id {}'.format(len(obj_matches), obj_id))

    else:
        # We found one match, as desired
        return obj_matches[0]


class SimulationRun:
    # Class for conducting a single simulation run with fixed parameters
    def __init__(
            self, trip_data_df: pd.DataFrame, chg_plan_df: pd.DataFrame | None,
            chargers_df: pd.DataFrame, depot_df: pd.DataFrame,
            vehicles_df: pd.DataFrame, deadhead_df: pd.DataFrame = None,
            ignore_deadhead: bool = False
    ):
        """
        Constructor for simulation run class
        """
        MIN_CHARGE_TIME = 1
        self.trip_data_df = trip_data_df.reset_index()
        self.trip_data_df['block_id'] = self.trip_data_df['block_id'].astype(
            str)
        self.trip_data_df.set_index(['block_id', 'trip_idx'], inplace=True)
        if chg_plan_df is not None:
            self.chg_plan_df = chg_plan_df.reset_index()
            self.chg_plan_df = self.chg_plan_df[
                self.chg_plan_df['chg_time'] > MIN_CHARGE_TIME
            ].copy()
            self.chg_plan_df['block_id'] = self.chg_plan_df['block_id'].astype(
                str
            )
            self.chg_plan_df.set_index(
            ['block_id', 'trip_idx', 'charger'], inplace=True)
        else:
            # Initialize an empty charge plan
            self.chg_plan_df = pd.DataFrame(
                columns=['block_id', 'trip_idx', 'charger', 'chg_time']
            ).set_index(['block_id', 'trip_idx', 'charger'])
        self.chargers_df = chargers_df.copy()
        self.depot_df = depot_df.copy()
        self.vehicles_df = vehicles_df.copy()
        self.ignore_deadhead = ignore_deadhead

        if deadhead_df is not None:
            self.deadhead_df = deadhead_df.copy()

        else:
            trip_start_locs = list(
                zip(self.trip_data_df['start_lat'].tolist(),
                    self.trip_data_df['start_lon'].tolist())
            )
            trip_end_locs = list(
                zip(self.trip_data_df['end_lat'].tolist(),
                    self.trip_data_df['end_lon'].tolist())
            )
            charger_locs = list(
                zip(
                    self.chargers_df['lat'].tolist(),
                    self.chargers_df['lon'].tolist()
                )
            )
            depot_coords = list(
                zip(
                    self.depot_df['lat'].tolist(),
                    self.depot_df['lon'].tolist()
                )
            )
            # Build up DH dict ourselves
            dh_dict = get_dh_dict(
                trip_start_locs=trip_start_locs,
                trip_end_locs=trip_end_locs,
                charger_locs=charger_locs,
                depot_coords=depot_coords
            )

            df = pd.DataFrame.from_dict(dh_dict, orient='index')
            df.index.set_names(['orig', 'dest'], inplace=True)
            df = df.reset_index()
            df[['orig_lat', 'orig_lon']] = pd.DataFrame(df['orig'].tolist())
            df[['dest_lat', 'dest_lon']] = pd.DataFrame(df['dest'].tolist())
            df = df.set_index(
                ['orig_lat', 'orig_lon', 'dest_lat', 'dest_lon']
            ).drop(columns=['orig', 'dest'])
            self.deadhead_df = df

        self.vehicles = list()
        # Initialize vehicles
        vids = self.vehicles_df.index.tolist()
        for v in vids:
            self.vehicles.append(
                Vehicle(
                    id=str(v),
                    min_chg=self.vehicles_df.loc[v, 'min_kwh'],
                    max_chg=self.vehicles_df.loc[v, 'max_kwh']
                )
            )

        # Initialize chargers
        self.chargers = list()
        chg_sites = self.chargers_df.index.tolist()
        for c in chg_sites:
            self.chargers.append(
                ChargingStation(
                    id=c, power=self.chargers_df.loc[c, 'kw'] / 60,
                    num_chargers=self.chargers_df.loc[c, 'n_chargers']
                )
            )
        self.calendar = Calendar()

        # Attributes that track outputs
        self.delay = dict()
        self.exog_delay = 0
        self.total_delay = 0
        self.queue_delay = dict()
        self.total_queue_delay = 0
        self.queue_delay_per_station = dict()
        self.rec_time = dict()
        self.plugin_times = dict()
        self.chargers_used = dict()
        self.charge_times = dict()
        self.charger_arrivals = dict()
        self.charge_amts = dict()
        self.actual_start_times = dict()
        self.actual_end_times = dict()
        self.total_recovery = 0
        self.unplanned_chgs = 0
        self.n_dead_batteries = 0
        self.n_missed_trips = 0
        self.total_chgs = 0
        self.total_chg_time = 0
        self.pct_trips_delayed = 0
        self.charges_df = None
        self.trip_times_df = None

    def copy(self):
        return SimulationRun(
            trip_data_df=self.trip_data_df,
            chg_plan_df=self.chg_plan_df,
            chargers_df=self.chargers_df,
            depot_df=self.depot_df,
            vehicles_df=self.vehicles_df,
            deadhead_df=self.deadhead_df,
            ignore_deadhead=self.ignore_deadhead
        )

    def look_up_deadhead(self, lat1, lon1, lat2, lon2, ignore_deadhead=None):
        if ignore_deadhead is None:
            ignore_deadhead = self.ignore_deadhead

        if ignore_deadhead:
            dh = {
                'distance': 0,
                'duration': 0
            }
        else:
            dh = self.deadhead_df.loc[(lat1, lon1, lat2, lon2)]
        return dh

    def check_charging_needed(
            self, veh, t, kwh_per_mi=None, method='basic'
    ):
        """
        Check whether charging is needed to complete next trip
        :param veh: Vehicle v under study
        :param t: Trip t under study
        :return: True if charging needed, False otherwise
        """
        v = veh.id
        if method == 'basic':
            if veh.chg < veh.min_chg:
                return True
            else:
                return False

        # Tolerance for negative charge
        # Set relatively high
        eps = 5

        # Calculate needed energy
        this_trip = self.trip_data_df.loc[(v, t)]
        if kwh_per_mi is None:
            energy_rate = self.trip_data_df.xs(key=v, level='block_id')[
                'kwh_per_mi'].mean()
        else:
            energy_rate = kwh_per_mi
        trip_dist = this_trip['total_dist']
        all_charger_dists = {
            s.id: self.look_up_deadhead(
                this_trip['end_lat'], this_trip['end_lon'],
                self.chargers_df.loc[s.id, 'lat'],
                self.chargers_df.loc[s.id, 'lon']
            )['distance']
            for s in self.chargers
        }
        min_charger_dist = min(all_charger_dists.values())

        # TODO: be more careful with this calc, may not be tight enough
        if t == 0 or (v, t+1) not in self.trip_data_df.index:
            # Initial depot trips have artificially high distances, set
            # to 0. Return trips can charge at base, so also set to 0.
            next_dist = 0

        else:
            next_trip = self.trip_data_df.loc[(v, t+1)]
            inter_trip_dist = self.look_up_deadhead(
                this_trip['end_lat'], this_trip['end_lon'],
                next_trip['start_lat'], next_trip['start_lon']
            )['distance']

            next_trip_dist = inter_trip_dist + trip_dist
            next_dist = min(min_charger_dist, next_trip_dist)

        chg_after = veh.chg - (trip_dist + next_dist) * energy_rate

        veh = get_object_by_id(v, self.vehicles)
        if chg_after < veh.min_chg - eps:
            return True
        else:
            return False

    def set_charging_plan(self, v, t, avg_e_rate, buffer=0):
        """
        After determining that charging is required, choose where and
        how much to charge.

        :param v: vehicle ID
        :param t: trip number
        :param chg: current charge level
        :param avg_e_rate: expected energy consumption across trips
            in kWh/mile
        :param buffer: additional charge added beyond expected need
        :return:
        """
        if self.ignore_deadhead:
            # Can have issues here when DH is ignored. Current code allows
            #   buses to use any charger regardless of how far away it is,
            #   when ignoring deadhead really should still require being
            #   within a certain distance to use a charger. So, we just
            #   throw an error.
            raise ValueError(
                'Cannot create a new charging plan when ignore_deadhead is True'
            )
        vid = v.id
        chg = v.chg
        # Select charging amount
        trips_reset = self.trip_data_df.reset_index()
        trips_left = trips_reset[
            (trips_reset['block_id'] == vid) & (trips_reset['trip_idx'] > t)
        ]

        req_chg = avg_e_rate * (
            trips_left['total_dist'].sum() + trips_left['dh_dist'].sum()
        )
        addl_chg = v.min_chg + req_chg + buffer - chg

        # Can't charge beyond vehicle's battery capacity
        max_cap = v.max_chg
        addl_chg = min(addl_chg, max_cap - chg)

        if addl_chg < 0:
            raise ValueError('Calculated negative additional charge required')

        # Select the charger that gets the job done fastest
        current_trip = self.trip_data_df.loc[(vid, t)]
        next_trip = self.trip_data_df.loc[(vid, t)]
        times = dict()
        to_and_from_times = dict()
        for s in self.chargers:
            charger_lat = self.chargers_df.loc[s.id, 'lat']
            charger_lon = self.chargers_df.loc[s.id, 'lon']
            to_charger_time = self.look_up_deadhead(
                current_trip['end_lat'], current_trip['end_lon'],
                charger_lat, charger_lon, ignore_deadhead=False
            )['duration']
            from_charger_time = self.look_up_deadhead(
                charger_lat, charger_lon,
                next_trip['start_lat'], next_trip['start_lon'],
                ignore_deadhead=False
            )['duration']
            chg_time = addl_chg / s.power
            to_and_from_times[s.id] = to_charger_time + from_charger_time
            times[s.id] = to_charger_time + from_charger_time \
                + s.get_queue_time() + chg_time

        if self.ignore_deadhead:
            min_time_charger = min(to_and_from_times, key=to_and_from_times.get)
            min_time = min(to_and_from_times.values())
            if min_time > 10:
                raise Warning(
                    'Scheduled a charge at a site over 10 minutes away '
                    '({:.2f} minutes round trip) when ignoring deadhead.'.format(
                        min_time
                    )
                )
        else:
            min_time_charger = min(times, key=times.get)

        # Update the charging plan for this vehicle so we don't charge
        # more than needed
        try:
            self.chg_plan_df = self.chg_plan_df.drop(
                labels=vid,
                axis='index',
                level='block_id'
            )
        except KeyError:
            # Some vehicles might not be in the charge plan at all, and
            # that is okay. There's nothing to remove.
            pass

        logger.debug(
            'Scheduled an unplanned charge on block {} at {} for {:.2f} minutes, '
            'adding {:.2f} kWh of charge to complete remaining {} trips.'.format(
                v.id, min_time_charger, chg_time, addl_chg, len(trips_left)
            )
        )
        return min_time_charger, addl_chg

    def calculate_exogenous_delay(self, method='sim'):
        # Evaluate baseline exogenous delay: the amount of delay that
        # would be present if buses didn't charge at all
        if method == 'calc':
            # This method calculates delays directly using Pandas, but
            # surprisingly it usually takes about twice as long as
            # just simulating the system with no charging. Hard to be
            # fast with the for loop I guess, and .xs() might be costing
            # some time too?
            method_1_start = time.time()
            self.exog_delay = 0
            for v in self.vehicles:
                v_df = self.trip_data_df.xs(key=v.id, level='block_id')[
                    ['start_time', 'start_lat', 'start_lon',
                     'end_lat', 'end_lon', 'duration']
                ].copy()
                if len(v_df) <= 1:
                    # If there's only one trip, it will never be delayed
                    continue
                v_df['dh_dest_lat'] = v_df['start_lat'].shift(-1)
                v_df['dh_dest_lon'] = v_df['start_lon'].shift(-1)
                # Merge in deadhead times
                v_df = v_df.merge(
                    self.deadhead_df['duration'].rename('dh_duration'),
                    left_on=['end_lat', 'end_lon', 'dh_dest_lat', 'dh_dest_lon'],
                    right_index=True,
                    how='left'
                )
                v_df['dh_duration'] = pd.to_timedelta(
                    v_df['dh_duration'].fillna(0),
                    unit='min'
                )
                # Set time bus is ready to start each trip
                v_df['ready_time'] = (
                    v_df['start_time'] + v_df['duration'] + v_df['dh_duration']
                ).shift(1).fillna(v_df['start_time'])

                v_df['delay'] = np.maximum(0, (v_df['ready_time'] - v_df[
                    'start_time']).dt.total_seconds() / 60)
                self.exog_delay += v_df['delay'].sum()

            logger.debug(
                'calculated exogenous delay: {}, took {:.3f} s'.format(
                    int(self.exog_delay),
                    time.time() - method_1_start
                )
            )

        elif method == 'sim':
            method_2_start = time.time()
            exog_sim = SimulationRun(
                trip_data_df=self.trip_data_df,
                chg_plan_df=None,
                chargers_df=self.chargers_df,
                depot_df=self.depot_df,
                vehicles_df=self.vehicles_df,
                deadhead_df=self.deadhead_df,
                ignore_deadhead=self.ignore_deadhead
            )
            exog_sim.trip_data_df['kwh_per_mi'] = 0
            exog_sim.run_sim()
            exog_sim.process_results()
            self.exog_delay = exog_sim.total_delay
            logger.debug(
                'simluated exogenous delay: {}, took {:.3f} s'.format(
                    int(self.exog_delay),
                    time.time() - method_2_start
                )
            )

        else:
            raise ValueError('method must be either "calc" or "sim"')

    def process_dead_battery(self, v, t):
        self.n_dead_batteries += 1
        # Identify number of remaining trips
        v_trips = self.trip_data_df.reset_index()[
            self.trip_data_df.reset_index()['block_id'] == v.id
            ]
        n_trips_left = len(v_trips[v_trips['trip_idx'] >= t])
        self.n_missed_trips += n_trips_left
        # Remove the last event from the calendar.
        self.calendar.remove_event()

    def run_sim(self):
        # Initialize calendar with first trip of all vehicles
        for v in self.vehicles:
            # Create event for first trip
            start_time = self.trip_data_df.loc[(v.id, 0), 'start_time']
            first_trip_start = TripStart(time=start_time, veh=v, trip=0)
            self.calendar.add_event(first_trip_start)

        it_ctr = 0
        # Proceed through calendar
        while len(self.calendar) > 0:
            it_ctr += 1

            # Select next event from calendar
            current_ev = self.calendar.get_next_event()

            if current_ev.type == 'trip_start':
                # Calculate delay and wait times for trip
                v = current_ev.veh
                t = current_ev.trip

                # Calculate 5% SOC -- note this is actually  5% of the
                # maximum charge (95% SoC by default)
                soc_5pct = (v.max_chg * 0.05)
                if v.chg < soc_5pct:
                    self.process_dead_battery(v, t)
                    continue

                self.actual_start_times[v.id, t] = current_ev.time

                sched_start = self.trip_data_df.loc[(v.id, t), 'start_time']
                time_diff = (sched_start - current_ev.time).total_seconds() / 60
                if time_diff >= -0.01:
                    # Ahead of schedule, so add to wait time
                    self.delay[v.id, t] = 0
                else:
                    self.delay[v.id, t] = -time_diff

                # Update vehicle charge
                e_rate = self.trip_data_df.loc[(v.id, t), 'kwh_per_mi']
                trip_dist = self.trip_data_df.loc[(v.id, t), 'total_dist']
                v.chg -= e_rate * trip_dist

                # Add end of trip to calendar
                t_end_time = current_ev.time + self.trip_data_df.loc[
                    (v.id, t), 'duration']
                new_ev = TripEnd(time=t_end_time, veh=v, trip=t)
                self.calendar.add_event(new_ev)

            elif current_ev.type == 'trip_end':
                v = current_ev.veh
                t = current_ev.trip

                self.actual_end_times[v.id, t] = current_ev.time

                if v.chg < 0:
                    self.process_dead_battery(v, t+1)
                    continue

                # Grab some key parameters
                trip_end_lat = self.trip_data_df.loc[(v.id, t), 'end_lat']
                trip_end_lon = self.trip_data_df.loc[(v.id, t), 'end_lon']
                vt_energy_rate = self.trip_data_df.loc[(v.id, t), 'kwh_per_mi']

                # Are any trips left on the schedule? If not, return to
                # base and do not add any new events.
                # TODO: re-evaluate how we handle depot trips
                if (v.id, t+1) not in self.trip_data_df.index:
                    self.calendar.remove_event()
                    continue

                # If a charge is scheduled at this time, drive to
                # the charger.
                charges = dict()
                for s in self.chargers:
                    try:
                        charges[s.id] = self.chg_plan_df.loc[
                            (v.id, t, s.id), 'chg_time'
                        ]
                    except KeyError:
                        # This KeyError may be raised if (1) we are on
                        # trip zero for some vehicle, which never has
                        # charging because it's just driving from the
                        # depot or (2) the key was not provided because
                        # the charging plan input only contains blocks
                        # that are expected to require charging. So we
                        # specify manually that no charging is planned
                        # in either case.
                        charges[s.id] = 0

                chg_planned = max(charges.values()) > 0

                if chg_planned:
                    chg_next = True
                    # Charging is scheduled for after this trip.
                    # Identify charging site, drive time, and
                    # charge required to drive there.
                    chg_site = max(charges, key=charges.get)
                    chg_time = charges[chg_site]

                else:
                    # Charging is not scheduled after this trip.
                    # Check if it needs to be done to complete the
                    # next trip.
                    chg_needed = self.check_charging_needed(
                        v, t+1, method='basic'
                    )

                    if chg_needed:
                        chg_next = True
                        self.unplanned_chgs += 1
                        # Charging must be done. Choose where and
                        # how much.
                        avg_energy_rate = self.trip_data_df.xs(
                            key=v.id, level='block_id'
                        )['kwh_per_mi'].mean()
                        chg_site, chg_amt = self.set_charging_plan(
                            v, t, avg_energy_rate)
                        chg_time = chg_amt / (self.chargers_df.loc[
                            chg_site, 'kw'] / 60)

                    else:
                        chg_next = False

                if chg_next:
                    # Next step is to charge (scheduled or not)
                    chg_site_lat = self.chargers_df.loc[chg_site, 'lat']
                    chg_site_lon = self.chargers_df.loc[chg_site, 'lon']
                    dh = self.look_up_deadhead(
                        trip_end_lat, trip_end_lon, chg_site_lat, chg_site_lon
                    )
                    time_to_site = datetime.timedelta(
                        minutes=dh['duration']
                    )
                    dist_to_site = dh['distance']
                    charger_arrival_time = current_ev.time + time_to_site
                    chg_used = self.trip_data_df.loc[
                                   (v.id, t), 'kwh_per_mi'
                               ] * dist_to_site
                    v.chg -= chg_used

                    # Ensure that we aren't going to exceed the maximum
                    # battery charge. If we are, reduce the charging
                    # duration.
                    chg_pwr = get_object_by_id(chg_site, self.chargers).power
                    if v.chg + chg_pwr * chg_time > v.max_chg:
                        # Calculate time to reach max charge
                        max_kwh_gain = v.max_chg - v.chg
                        chg_time = max_kwh_gain / chg_pwr

                        if max_kwh_gain < 0:
                            # uncommon edge case: we could be above
                            # nominal capacity due to regen braking. if
                            # set, just set charge time to zero. note
                            # that bus will still go the charger and
                            # queue if necessary, but immediately
                            # finish charging. not very realistic, but
                            # a very rare case.
                            logger.info(
                                'Negative charge time found for block {}, trip '
                                '{}. Current battery level is {} and maximum is '
                                '{}. Last trip used {} kWh/mi.'.format(
                                    v.id, t, v.chg, v.max_chg,
                                    self.trip_data_df.loc[
                                        (v.id, t), 'kwh_per_mi']
                                )
                            )
                            chg_time = 0

                    new_ev = ChargerArrival(
                        time=charger_arrival_time, veh=v, trip=t,
                        chg_time=chg_time, chg_site=chg_site)
                    self.calendar.add_event(new_ev)

                else:
                    # No need to charge. Move to next trip.
                    next_trip = self.trip_data_df.loc[
                        (v.id, t + 1)
                    ]
                    next_trip_dh = self.look_up_deadhead(
                        trip_end_lat, trip_end_lon,
                        next_trip['start_lat'], next_trip['start_lon']
                    )

                    time_to_next = datetime.timedelta(
                        minutes=next_trip_dh['duration']
                    )
                    # Next trip starts either when we get there, or when
                    # scheduled
                    ready_time = current_ev.time + time_to_next
                    sched_start = self.trip_data_df.loc[
                        (v.id, t+1), 'start_time'
                    ]

                    next_start_time = max(ready_time, sched_start)
                    self.rec_time[v.id, t+1] = max(
                        0., (sched_start - ready_time).total_seconds() / 60
                    )
                    next_start_dist = next_trip_dh['distance']
                    v.chg -= next_start_dist * vt_energy_rate
                    new_ev = TripStart(
                        time=next_start_time, veh=v, trip=t+1
                    )
                    self.calendar.add_event(new_ev)

            elif current_ev.type == 'chg_arr':
                chg_site = get_object_by_id(current_ev.chg_site, self.chargers)
                v = current_ev.veh
                t = current_ev.trip
                if v.chg < 0:
                    self.process_dead_battery(v, t+1)
                    continue

                chg_req = ChargeRequest(
                    time_made=current_ev.time, veh=v, trip=current_ev.trip,
                    chg_site=chg_site, chg_time=current_ev.chg_time)
                self.charger_arrivals[v.id, current_ev.trip] = current_ev.time
                self.chargers_used[v.id, current_ev.trip] = current_ev.chg_site
                self.charge_times[v.id, current_ev.trip] = current_ev.chg_time
                self.charge_amts[v.id, current_ev.trip] = \
                    chg_site.power * current_ev.chg_time

                if chg_site.is_full():
                    # If charger is full, add the arriving vehicle to
                    # the queue
                    chg_site.add_to_queue(chg_req)

                else:
                    # If charger is available, start charging and add
                    # event for charge completion to calendar.
                    chg_site.start_charging(chg_req)
                    chg_end_time = current_ev.time + datetime.timedelta(
                        minutes=current_ev.chg_time
                    )
                    new_ev = ChargerDeparture(
                        time=chg_end_time, veh=v, trip=current_ev.trip,
                        chg_site=current_ev.chg_site,
                        chg_time=current_ev.chg_time)
                    self.plugin_times[v.id, current_ev.trip] = current_ev.time
                    # No queue delay is incurred because this vehicle
                    # never enters a station queue
                    self.queue_delay[
                        v.id, current_ev.trip, current_ev.chg_site] = 0
                    self.calendar.add_event(new_ev)

            elif current_ev.type == 'chg_dpt':
                v = current_ev.veh
                t = current_ev.trip
                s = current_ev.chg_site

                chg_site = get_object_by_id(s, self.chargers)

                try:
                    chg_site.finish_charging(v.id)
                except ValueError:
                    raise ValueError('Cannot remove vehicle that is not'
                                     'charging.')

                # Track outputs
                self.total_chgs += 1
                self.total_chg_time += current_ev.chg_time

                # Advance queue. If a new vehicle starts charging, we
                # need to create an event for its charge completion.
                new_req = chg_site.advance_queue()
                if new_req is not None:
                    # Track how long this request waited for
                    self.queue_delay[new_req.veh.id, new_req.trip, s] = \
                        (current_ev.time - new_req.time_made).total_seconds() / 60
                    self.plugin_times[new_req.veh.id, new_req.trip] = \
                        current_ev.time
                    # Process new charging request, add departure to
                    # calendar.
                    req_chg_end = current_ev.time + datetime.timedelta(
                        minutes=new_req.chg_time
                    )
                    new_dept = ChargerDeparture(
                        time=req_chg_end, veh=new_req.veh, trip=new_req.trip,
                        chg_site=s, chg_time=new_req.chg_time)
                    self.calendar.add_event(new_dept)

                # Update battery level
                v.chg += current_ev.chg_time * chg_site.power
                if v.chg > v.max_chg:
                    logger.info(
                        'Battery level of bus {} is {:.1f} kWh, exceeding its'
                        ' maximum of {:.1f} kWh'.format(
                            v.id, v.chg, v.max_chg
                        )
                    )

                # Move vehicle to start of next trip
                next_trip = self.trip_data_df.loc[(v.id, t+1)]
                this_charger = self.chargers_df.loc[s]
                dh = self.look_up_deadhead(
                    this_charger['lat'], this_charger['lon'],
                    next_trip['start_lat'], next_trip['start_lon']
                )

                dist_to_start = dh['distance']
                time_to_start = datetime.timedelta(
                    minutes=dh['duration']
                )
                v.chg -= dist_to_start * self.trip_data_df.loc[
                    (v.id, t), 'kwh_per_mi'
                ]

                next_start_time = max(
                    current_ev.time + time_to_start,
                    self.trip_data_df.loc[(v.id, t+1), 'start_time']
                )
                new_ev = TripStart(time=next_start_time, veh=v, trip=t+1)
                self.calendar.add_event(new_ev)

            else:
                raise AttributeError('Unrecognized or absent event type:'
                                     '{}'.format(current_ev.type))

            # Remove completed event from calendar
            self.calendar.remove_event()

    def check_battery_level(self, veh, t):
        # TODO: set this based on battery capacity and SoC
        abs_min_soc = 5

        if veh.chg < abs_min_soc:
            self.n_dead_batteries += 1
            # Identify number of remaining trips
            v_trips = self.trip_data_df[
                self.trip_data_df['block_id'] == veh.id
            ]
            n_trips_left = len(v_trips[v_trips['trip_idx'] > t])
            self.n_missed_trips += n_trips_left

    def process_results(self):
        # Total delay
        self.total_delay = sum(self.delay.values())
        # Add penalty for all missed trips (at least 60 min)
        max_delay = max(60, max(self.delay.values()))
        self.total_delay += self.n_missed_trips * max_delay

        # Total queue delay
        self.total_queue_delay = sum(self.queue_delay.values())

        # Queue delay per station
        # Results for each queue
        q_sites = set([s for (v, t, s) in self.queue_delay])
        for s in q_sites:
            q_vals = {(v, t): self.queue_delay[v, t, s2]
                      for (v, t, s2) in self.queue_delay if s2 == s}
            times_in_q = np.array(list(q_vals.values()))
            self.queue_delay_per_station[s] = times_in_q.tolist()

        # % trips delayed
        all_delays = list(self.delay.values())
        self.pct_trips_delayed = len([d for d in all_delays if d > 5]) / len(
            all_delays) * 100

        # Total recovery time
        self.total_recovery = sum(self.rec_time.values())

        # Charging behavior DF
        self.charges_df = pd.DataFrame(
            data={
                'charger': self.chargers_used,
                'arrival_time': self.charger_arrivals,
                'plugin_time': self.plugin_times,
                'chg_time': self.charge_times,
                'chg_kwh': self.charge_amts
            }
        )
        if self.charges_df.empty:
            self.charges_df.index = pd.MultiIndex(
                levels=[[], []], codes=[[], []],
                names=['block_id', 'trip_id']
            )
        else:
            self.charges_df.index.set_names(
                ['block_id', 'trip_idx'], inplace=True
            )

        self.charges_df.to_csv('test_results.csv')

        # Trip timing DF
        self.trip_times_df = pd.DataFrame(
            data={
                'actual_start_time': self.actual_start_times,
                'actual_end_time': self.actual_end_times,
                'delay': self.delay
            }
        )
        self.trip_times_df.index.set_names(['block_id', 'trip_idx'], inplace=True)
        self.trip_times_df.to_csv('test_trip_times.csv')

    def print_results(self):
        logger.info('Total recovery time: {:.2f} minutes'.format(
            self.total_recovery))
        logger.info('Total delay: {:.2f} minutes'.format(self.total_delay))
        logger.info('Maximum delay: {:.2f} minutes'.format(max(self.delay.values())))
        logger.info('Percentage of trips delayed over 5 minutes: {:.2f}%'.format(
            self.pct_trips_delayed))
        logger.info('Total queue waiting time: {:.2f} minutes'.format(
            sum(self.queue_delay.values())))
        logger.info('Total number of charger visits: {}'.format(self.total_chgs))
        logger.info('Number of unscheduled charges: {}'.format(self.unplanned_chgs))
        logger.info('Total charging time: {:.2f} minutes'.format(
            self.total_chg_time))
        logger.info('Average time per charge: {:.2f} minutes'.format(
            self.total_chg_time / self.total_chgs))

        q_sites = set([s for (v, t, s) in self.queue_delay])
        for s in q_sites:
            logger.info('Mean time in queue for {}: {} minutes'.format(s,
                  np.mean(self.queue_delay_per_station[s])))


class SimulationBatch:
    # Class for conducting a series of independent simulation runs with
    # varying data
    def __init__(
            self, trip_data_df: pd.DataFrame, chg_plan_df: pd.DataFrame | None,
            chargers_df: pd.DataFrame, depot_df: pd.DataFrame,
            vehicles_df: pd.DataFrame, n_sims: int,
            deadhead_df: pd.DataFrame = None, seed: int = None,
            ignore_deadhead: bool = False, vary_duration: bool = True,
            vary_energy: bool = True
    ):
        # Create a baseline simulation instance
        self.base_sim = SimulationRun(
            trip_data_df=trip_data_df,
            chg_plan_df=chg_plan_df,
            chargers_df=chargers_df,
            depot_df=depot_df,
            vehicles_df=vehicles_df,
            deadhead_df=deadhead_df,
            ignore_deadhead=ignore_deadhead
        )

        self.vary_duration = vary_duration
        self.vary_energy = vary_energy
        self.n_sims = n_sims
        self.delay = np.zeros(n_sims)
        self.exog_delay = np.zeros(n_sims)
        self.charging_delay = np.zeros(n_sims)
        self.pct_trips_delayed = np.zeros(n_sims)
        self.n_unplanned_charges = np.zeros(n_sims)
        self.n_dead_batteries = np.zeros(n_sims)
        self.n_missed_trips = np.zeros(n_sims)

        if seed is not None:
            # Instantiate random number generator
            self.rng = np.random.default_rng(seed)

        else:
            self.rng = np.random.default_rng()

    def run(self):
        if self.vary_energy:
            energy_loc = self.base_sim.trip_data_df['kwh_per_mi_mean']
            energy_scale = self.base_sim.trip_data_df['kwh_per_mi_err']

        if self.vary_duration:
            # Read in schedule deviation info
            dur_df = pd.read_csv(
                '../data/processed/schedule_deviation_distributions.csv',
                parse_dates=['date'],
                dtype={'trip_id': str, 'route': str}
            )
            dur_df = dur_df[
                dur_df['route'].isin(self.base_sim.trip_data_df['route'].unique())
            ].copy()
            dur_gb = dur_df.groupby('route')

        if not self.vary_duration:
            self.base_sim.calculate_exogenous_delay()

        for n in range(self.n_sims):
            sim = self.base_sim.copy()
            if self.vary_energy:
                sim.trip_data_df['kwh_per_mi'] = \
                    self.rng.normal(loc=energy_loc, scale=energy_scale)
            if self.vary_duration:
                sample = True
                if sample:
                    # Randomly sample
                    sim.trip_data_df['duration'] = pd.to_timedelta(
                        sim.trip_data_df.apply(
                            lambda rw: rw['duration_sched'] * (
                                1 + dur_gb.get_group(
                                    rw['route'])['duration_difference_pct'].sample(
                                        n=1).tolist()[0] / 100
                            ),
                            axis=1
                        ),
                        unit='min'
                    )

                else:
                    sim.trip_data_df['duration'] = pd.to_timedelta(
                        sim.trip_data_df['duration_mean']
                    )
                    # Calculate exogenous delay
                    sim.calculate_exogenous_delay()
                    self.exog_delay[n] = sim.exog_delay

            else:
                # We only need to get the exogenous delay once, from
                # the base simulation, if duration isn't varied
                self.exog_delay[n] = self.base_sim.exog_delay

            sim.run_sim()
            sim.process_results()
            self.delay[n] = sim.total_delay
            self.charging_delay[n] = sim.total_delay - self.exog_delay[n]
            self.pct_trips_delayed[n] = sim.pct_trips_delayed
            self.n_unplanned_charges[n] = sim.unplanned_chgs
            self.n_dead_batteries[n] = sim.n_dead_batteries
            self.n_missed_trips[n] = sim.n_missed_trips

    def process_results(self):
        delay_mean = np.mean(self.delay)
        logger.info('Mean total delay: {:.2f} minutes'.format(delay_mean))

        induced_mean = np.mean(self.charging_delay)
        logger.info('Mean charging-induced delay: {:.2f} minutes'.format(induced_mean))
        delay_std = np.std(self.delay, ddof=1)
        # logger.info('Standard deviation: {:.2f} minutes'.format(delay_std))
        # Calculate 95% confidence interval
        alpha = 0.05
        t_val = tstat.ppf(1-alpha/2, self.n_sims-1)
        half_len = t_val*delay_std/np.sqrt(self.n_sims)
        logger.info('95% confidence interval on mean delay: '
                     '[{:.2f}, {:.2f}]'.format(
            delay_mean - half_len, delay_mean + half_len)
        )





