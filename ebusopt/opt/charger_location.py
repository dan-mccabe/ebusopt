import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
from matplotlib import ticker
import pickle
import logging
import time
import copy
from pathlib import Path
from pyomo.environ import ConcreteModel, Set, Var, Binary, Constraint, \
    Objective, SolverFactory, value, NonNegativeIntegers, NonNegativeReals, \
    SolverStatus, TerminationCondition
from ebusopt.gtfs_beb import get_updated_osm_data
# Suppress matplotlib log (gets annoying at DEBUG level)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('pyomo').setLevel(logging.WARNING)
# Set up logger for this module
logger = logging.getLogger('charger_location')


class ChargerLocationModel:
    def __init__(self, vehicles, veh_trip_pairs, chg_sites, chg_lims,
                 trip_start_times, trip_end_times, trip_dists,
                 inter_trip_dists, trip_start_chg_dists, trip_end_chg_dists,
                 trip_start_chg_times, trip_end_chg_times, inter_trip_times,
                 chg_rates, energy_rates, site_costs, charger_costs,
                 max_chargers, zero_time, depot_coords, trips_df):
        """
        Initialize BEB Optimal Charger Location model.

        :param vehicles: list of vehicle IDs
        :param veh_trip_pairs: list of (vehicle ID, trip index) tuples
        :param chg_sites: list of charging site names
        :param chg_lims: dict of upper charge limits of each bus
        :param trip_start_times: dict of trip start times
        :param trip_end_times: dict of trip end times
        :param trip_dists: dict of trip distances in miles
        :param inter_trip_dists: dict of trip deadhead distances in miles
        :param trip_start_chg_dists: dict of deadhead distances to chargers
        :param trip_end_chg_dists: dict of deadhead distances from chargers
        :param trip_start_chg_times: dict of deadhead times to chargers
        :param trip_end_chg_times: dict of deadhead times from chargers
        :param inter_trip_times: dict of deadhead times between trips
        :param chg_rates: dict of charger power levels in kWh/min
        :param energy_rates: dict of energy consumption rates per trip
        :param site_costs: dict giving cost of each charging site
        :param charger_costs: dict giving cost of chargers at each site
        :param max_chargers: dict giving max number of chargers at each site
        :param zero_time: zero offset time as float
        :param depot_coords: coordinates of depot as tuple
        :param trips_df: DataFrame of all necessary trip attributes
        """
        super().__init__()
        # Set attributes
        self.vehicles = vehicles
        self.charging_vehs = list()
        self.infeas_vehs = list()
        self.veh_trip_pairs = veh_trip_pairs
        self.trips_per_veh = {
            v: sorted([t for (vv, t) in self.veh_trip_pairs if vv == v])
            for v in self.vehicles}
        self.charging_vts = veh_trip_pairs
        self.chg_sites = sorted(chg_sites)
        self.chg_lims = chg_lims
        self.trip_start_times = trip_start_times
        self.trip_end_times = trip_end_times
        self.trip_dists = trip_dists
        self.inter_trip_dists = inter_trip_dists
        self.trip_start_chg_dists = trip_start_chg_dists
        self.trip_end_chg_dists = trip_end_chg_dists
        self.trip_start_chg_times = trip_start_chg_times
        self.trip_end_chg_times = trip_end_chg_times
        self.inter_trip_times = inter_trip_times
        self.depot_coords = depot_coords
        # Set of possible charger power levels at each site
        self.chg_rates = chg_rates
        # Maximum charge rate available at each site (used to check
        # feasibility)
        self.energy_rates = energy_rates
        self.site_costs = site_costs
        self.charger_costs = charger_costs
        self.max_chargers = max_chargers
        self.zero_time = zero_time
        self.MAX_CHG_TIME = 45
        self.block_energy_needs = dict()
        self.backup_trips = dict()
        self.conflict_sets = None
        self.last_feas_trips = dict()
        # Upper bound on charging time to avoid delays
        self.chg_time_avail = dict()
        self.s_vt = dict()
        self.set_max_chg_time()
        # TODO: restructure inputs to this class now that trips DF is
        #   contained within it
        self.trips_df = trips_df

        # Initialize result attributes
        self.model = None
        self.alpha = None
        self.solver_status = None
        self.soln_time = None
        self.n_backups = 0
        self.opt_cost = None
        self.capital_costs = None
        self.opt_stations = None
        self.num_chargers = None
        self.opt_powers = None
        self.opt_charges = None
        self.opt_total_chg_time = None
        self.opt_deadhead_time = None
        self.chg_ratios = None
        self.chg_intervals = None
        self.chgs_per_veh = None
        self.interarrival_times = None
        self.service_times = None
        self.chg_schedule = None
        self.prior_chgs = None

    def set_max_chg_time(self):
        """
        Set self.chg_time_avail based on trip start/end times
        """
        for (v, t) in self.charging_vts:
            for s in self.chg_sites:
                if (v, t+1) in self.charging_vts:
                    self.chg_time_avail[v, t, s] = max(
                        0, self.trip_start_times[v, t+1]
                        - self.trip_end_times[v, t]
                        - self.trip_end_chg_times[v, t, s]
                        - self.trip_start_chg_times[v, t+1, s])
                else:
                    # Allow charging up to max duration at end of day
                    self.chg_time_avail[v, t, s] = self.MAX_CHG_TIME

        self.chg_time_avail = {
            k: min(self.chg_time_avail[k], self.MAX_CHG_TIME)
            for k in self.chg_time_avail}

    def check_charging_needs(self):
        """Filter out blocks that don't require charging."""
        trip_energy_needs = dict()
        for (v, t) in self.veh_trip_pairs:
            try:
                trip_energy_needs[v, t] = self.energy_rates[v, t] * (
                        self.trip_dists[v, t] + self.inter_trip_dists[v, t])
            except KeyError:
                raise KeyError(
                    'Key error for vehicle {}, trip {}'.format(v, t))

        # Identify which vehicles can be excluded because they never
        # need to charge
        trip_energy_needs = {(v, t): self.energy_rates[v, t] * (
                self.trip_dists[v, t] + self.inter_trip_dists[v, t])
                             for (v, t) in self.veh_trip_pairs}
        self.block_energy_needs = dict()
        self.charging_vehs = list()
        for v in self.vehicles:
            ch = self.chg_lims[v]
            empty_flag = False
            for t in self.trips_per_veh[v]:
                ch -= trip_energy_needs[v, t]
                if ch < 0 and not empty_flag:
                    empty_flag = True
                    self.charging_vehs.append(v)
                    self.last_feas_trips[v] = t-1
            self.block_energy_needs[v] = self.chg_lims[v] - ch

        logger.info('Number of blocks that require charging: {}'.format(
            len(self.charging_vehs)))

        self.charging_vts = [(v, t) for (v, t) in self.veh_trip_pairs
                             if v in self.charging_vehs]
        logger.info('Number of trips in charging blocks: {}'.format(
            len(self.charging_vts)))

    def check_feasibility(self):
        """
        Check whether it is feasible to perform necessary charging
        without delays. If not, report the vehicles for which charging
        is infeasible. Remove them from the optimization analysis.
        """
        self.infeas_vehs = list()
        for v in self.charging_vehs:
            if v == 5654025:
                pass
            ch = self.chg_lims[v]
            trips = sorted(t for (vv, t) in self.charging_vts if v == vv)
            for t in trips:
                # Subtract energy needed to complete trip
                ch = ch - self.energy_rates[v, t] * self.trip_dists[v, t]
                # What's the maximum charge gain possible?
                next_dist = {s: self.trip_start_chg_dists[v, t+1, s]
                             if (v, t+1) in self.charging_vts else 0
                             for s in self.chg_sites}
                gain_dict = {
                    s: self.chg_time_avail[v, t, s] * self.chg_rates[s]
                    - self.energy_rates[v, t] * (
                               self.trip_end_chg_dists[v, t, s] + next_dist[s])
                    for s in self.chg_sites}

                for s in self.chg_sites:
                    # Confirm that we don't exceed the battery capacity
                    min_ch = ch - self.energy_rates[v, t] \
                             * self.trip_end_chg_dists[v, t, s]
                    if min_ch < 0:
                        gain_dict[s] = -self.chg_lims[v]
                    max_ch = min_ch + self.chg_time_avail[v, t, s] \
                             * self.chg_rates[s]
                    # Ensure bus only charges to battery capacity
                    if max_ch > self.chg_lims[v]:
                        gain_dict[s] -= max_ch - self.chg_lims[v]

                max_gain = max(gain_dict.values())
                max_gain_s = max(gain_dict, key=gain_dict.get)

                # How much charge is lost driving to the next site?
                no_chg_loss = -self.inter_trip_dists[v, t] * self.energy_rates[
                    v, t]
                # If chargers are far enough away that we lose more
                # charge than we would by not charging, just go to the
                # next trip.
                if max_gain < no_chg_loss:
                    max_gain = no_chg_loss
                    min_ch = ch - no_chg_loss
                else:
                    min_ch = ch - self.energy_rates[v, t] \
                             * self.trip_end_chg_dists[v, t, max_gain_s]

                # Do we run out of battery?
                if min_ch < 0:
                    self.infeas_vehs.append(v)
                    break
                else:
                    # Update charge
                    ch = ch + max_gain

        logger.info('Number of infeasible blocks: {}'.format(
            len(self.infeas_vehs)))
        if self.infeas_vehs:
            logger.info('Infeasible block IDs: {}'.format(self.infeas_vehs))

    def run_brp_heuristic(
            self, infeas_vts, dh_dist, bu_kwh, a_set,
            method='random', energy_rate=3):
        """
        Heuristic method for dispatching backup buses. Replaces some
        trips of infeasible original blocks so that they are all
        feasible without layover charging and assigns a backup bus to
        each of these trips, aiming to minimize the total number of
        backup buses used.

        :param infeas_vts: (block_id, trip_idx) tuples on infeasible
            blocks
        :param dh_dist: deadhead distance between all pairs of
            infeas_vts
        :param bu_kwh: energy consumption per mile of backup buses
        :param a_set: set of feasible arcs in BEB-BRP
        :param method: method for selecting next trip to serve with a
            backup bus.
        :return: backup blocks as dict, where keys are block numbers
            and values are lists of (v, t) pairs served on each backup
            block
        """
        # Copies so we don't modify inputs
        trips_left = copy.copy(infeas_vts)
        dh_dist = copy.copy(dh_dist)
        # Dummy trips
        src = (0, 0)
        sink = (0, 0)

        # Original blocks
        obs = self.infeas_vehs

        # Energy deficit of each OB
        self.trip_dists[src] = 0
        dh_dist[(*src, *sink)] = 0
        # TODO: it seems like dh_dist might not be correct? Doesn't seem
        #   to agree with ZEBRA
        arc_kwh = {
            (u, i, v, j): energy_rate * (
                    self.trip_dists[u, i] + dh_dist[u, i, v, j])
            for (u, i, v, j) in dh_dist}
        ob_trips = {
            u: sorted(t for (v, t) in trips_left if v == u) for u in obs}
        nrg_def = {
            u: arc_kwh[(*src, u, ob_trips[u][0])] + arc_kwh[
                (u, ob_trips[u][-1], *sink)
            ] + sum(arc_kwh[u, i, u, i + 1] for i in ob_trips[u][:-1])
            - self.chg_lims[u] for u in obs
        }

        # Energy savings from replacing a single trip
        nrg_svgs = dict()
        for u in obs:
            # Replacing middle trips
            for ix, j in enumerate(ob_trips[u]):
                # Get index of previous trip
                if ix == 0:
                    o = src[0]
                    i = src[1]
                else:
                    o = u
                    i = ob_trips[u][ix - 1]
                # Get index of next trip
                if ix == len(ob_trips[u]) - 1:
                    d = sink[0]
                    k = sink[1]
                else:
                    d = u
                    k = ob_trips[u][ix + 1]

                nrg_svgs[u, j] = energy_rate * (
                        dh_dist[o, i, u, j] + self.trip_dists[u, j]
                        + dh_dist[u, j, d, k] - dh_dist[o, i, d, k])

        ob_trips_left = copy.copy(ob_trips)
        bu_trips = list()
        for v in obs:
            while nrg_def[v] > 0:
                # Select a trip to replace
                if method == 'random':
                    next_trip = (v, int(np.random.choice(ob_trips_left[v])))

                elif method == 'semirandom':
                    if np.random.rand() < 0.1:
                        next_trip = (
                            v, int(np.random.choice(ob_trips_left[v])))
                    else:
                        next_trip = (v, ob_trips_left[v][0])

                elif method == 'lifo':
                    # Take the last trip
                    next_trip = (v, ob_trips_left[v][-1])

                else:
                    # Take the next trip
                    next_trip = (v, ob_trips_left[v][0])

                # Update trip coverage
                bu_trips.append(next_trip)
                ob_trips_left[v].remove(next_trip[1])
                nrg_def[v] -= nrg_svgs[next_trip]
                if nrg_def[v] <= 0:
                    continue

                # Update energy savings for replacing each remaining
                # trip
                nrg_svgs.pop(next_trip)
                j = next_trip[1]

                # Is there a trip before the current one?
                # Need to add back in next_trip here since it's
                # removed in block above
                v_trips = sorted(t for t in ob_trips_left[v] + [j])

                j_idx = v_trips.index(j)
                if j_idx == 0:
                    # We replaced the first trip. We only need to
                    # update the second trip.
                    k = (v, v_trips[j_idx + 1])
                    if j_idx + 2 in range(len(v_trips)):
                        l = (v, v_trips[j_idx + 2])
                    else:
                        l = sink
                    nrg_svgs[k] = energy_rate * (
                            dh_dist[(*src, *k)] + self.trip_dists[k]
                            + dh_dist[(*k, *l)] - dh_dist[(*src, *l)])

                elif j_idx == len(v_trips) - 1:
                    # We replaced the last trip. We only need to
                    # update the second-to-last trip.
                    if j_idx - 2 in range(len(v_trips)):
                        h = (v, v_trips[j_idx - 2])
                    else:
                        h = src
                    i = (v, v_trips[j_idx - 1])
                    nrg_svgs[i] = energy_rate * (
                            dh_dist[(*h, *i)] + self.trip_dists[i]
                            + dh_dist[(*i, *sink)]
                            - dh_dist[(*h, *sink)])
                else:
                    # We need to update both the previous and next trip.
                    i = (v, v_trips[j_idx - 1])
                    k = (v, v_trips[j_idx + 1])

                    if j_idx - 2 in range(len(v_trips)):
                        h = (v, v_trips[j_idx - 2])
                    else:
                        h = src

                    if j_idx + 2 in range(len(v_trips)):
                        l = (v, v_trips[j_idx + 2])
                    else:
                        l = sink

                    nrg_svgs[i] = energy_rate * (
                            dh_dist[(*h, *i)] + self.trip_dists[i]
                            + dh_dist[(*i, *k)] - dh_dist[(*h, *k)])
                    nrg_svgs[k] = energy_rate * (
                            dh_dist[(*i, *k)] + self.trip_dists[k]
                            + dh_dist[(*k, *l)] - dh_dist[(*i, *l)])

        logger.debug(
            'Number of trips served by backups: {}'.format(len(bu_trips)))
        # Assign backup trips to specific backup vehicles
        # First, sort by start time
        bu_trips_sorted = sorted(bu_trips, key=self.trip_start_times.get)
        bu_blocks = dict()
        bu_energy = dict()
        bu_idx = 0
        for k in bu_trips_sorted:
            # Assign to a block
            # Are there blocks this could be added to?
            # Time feasibility
            time_feas_blocks = list()
            k_arcs = [a for a in a_set if a[2:] == k]

            for a in k_arcs:
                for b in bu_blocks:
                    if a[:2] == bu_blocks[b][-1]:
                        time_feas_blocks.append(b)

            # Energy feasibility
            feas_blocks = [
                b for b in time_feas_blocks
                if bu_kwh - bu_energy[b] >= arc_kwh[(*bu_blocks[b][-1], *k)] \
                   + arc_kwh[k[0], k[1], sink[0], sink[1]
                ]
            ]
            if feas_blocks:
                block_ends = {
                    b: self.trip_end_times[bu_blocks[b][-1]]
                    for b in feas_blocks}
                best_block = max(block_ends, key=block_ends.get)
                # Update block and energy
                last_trip = bu_blocks[best_block][-1]
                # Add to the chosen block
                bu_blocks[best_block].append(k)
                # Update energy
                bu_energy[best_block] += arc_kwh[(*last_trip, *k)]

            else:
                # Create a new block that starts with this trip
                bu_blocks[bu_idx] = [k]
                bu_energy[bu_idx] = arc_kwh[(*src, *k)]
                bu_idx += 1

        for b in bu_blocks:
            bu_blocks[b].append(sink)

        return bu_blocks

    def get_compatible_trips(
            self, bu_vts, min_layover=0, max_layover=1000,
            depot_vt=(0, 0), dh_distance_fname=None):
        """
        For a set of trips on infeasible blocks, get the deadhead
        distances between them and the set of trips that are compatible
        to be served successively for running BEB-BRP.

        :param bu_vts: set of (v, t) tuples on infeasible blocks
        :param min_layover: minimum layover time required between trips
            (in minutes)
        :param max_layover: maximum layover time allowed between trips
            (in minutes)
        :param depot_vt: dummy (v, t) tuple representing the depot
        :param dh_distance_fname: filename of pickle file where DH data
            is saved for lookup
        :return: compat_arcs, list of compatible trip pairs (arcs)
            dh_dist, deadhead distances between trip pairs (in miles)
        """
        max_trip_idx = {v: max(t for (v2, t) in self.veh_trip_pairs if v2 == v)
                        for v in self.infeas_vehs}
        # Calculate deadhead distances and durations
        # First, read in trip data
        trips_df = self.trips_df.copy()
        # We add 1 to the trip index in the optimization models (so that
        # it starts at 0 rather than 1). The trips DF is still indexed
        # from 0. Here, we make a copy and add 1 so it's aligned with
        # the vehicle and trip indices in bu_vts.
        trips_df['trip_idx'] += 1
        trips_df = trips_df.set_index(['block_id', 'trip_idx'])
        # Filter down to just trips on infeasible blocks
        trips_df = trips_df.loc[bu_vts]
        # Get start and end coordinates of all trips (for deadhead lookup)
        end_coords = [self.depot_coords] + list(
            trips_df.apply(
                lambda x: (x['end_lat'], x['end_lon']), axis=1).unique())
        start_coords = [self.depot_coords] + list(
            trips_df.apply(
                lambda x: (x['start_lat'], x['start_lon']), axis=1).unique())

        start_srs = trips_df['start_time']
        end_srs = trips_df['end_time']

        if dh_distance_fname is None:
            dh_distance_fname = str(
                Path(__file__).resolve().parent.parent /
                'data' / 'osm' / 'osm_charge_data.pickle')
        if not dh_distance_fname:
            dh_distance_fname = 'data/osm/osm_charge_data.pickle'

        # Get DH data from OSM
        dh_by_coords = get_updated_osm_data(
            origins=end_coords, dests=start_coords,
            filename=dh_distance_fname)

        # Initialize DH distances (indexed by v-t pairs)
        dh_dist = dict()

        # We only need to calculate costs for trips that are compatible.
        # We can start by just looking at start/end times.
        compat_no_dh = list()
        for (u, i) in bu_vts:
            later_trips = start_srs[
                datetime.timedelta(minutes=min_layover)
                <= start_srs - end_srs.loc[(u, i)]
                ].index.tolist()
            not_too_late_trips = start_srs[
                start_srs - end_srs.loc[(u, i)]
                <= datetime.timedelta(minutes=max_layover)].index.tolist()
            good_trips = list(set(later_trips) & set(not_too_late_trips))

            same_block_trips = [(u, j) for j in range(i + 1, max_trip_idx[u])
                                if (u, j) not in good_trips]
            trips_to_add = [(u, i, v, j) for (v, j)
                            in good_trips + same_block_trips]
            compat_no_dh += trips_to_add

            # While we're here, establish distances for driving to and
            # from the depot.
            # Leaving: DH goes from depot to start of trip (u, i)
            # Where does next trip start?
            dest_loc = (trips_df.loc[(u, i), 'start_lat'],
                        trips_df.loc[(u, i), 'start_lon'])

            dh_params = dh_by_coords[self.depot_coords, dest_loc]
            dh_dist_od = dh_params['distance']

            # Index for backup blocks
            dh_dist[(*depot_vt, u, i)] = dh_dist_od

            # Index for original blocks
            dh_dist[(u, 0, u, i)] = dh_dist_od

            # Returning: DH goes from end of trip (u, i) to depot
            # Where does v, j end?
            src_loc = (trips_df.loc[(u, i), 'end_lat'],
                       trips_df.loc[(u, i), 'end_lon'])
            dh_params = dh_by_coords[src_loc, self.depot_coords]

            dh_dist_od = dh_params['distance']

            # Index for backup blocks
            dh_dist[(u, i, *depot_vt)] = dh_dist_od

            # Index for original blocks
            dh_dist[(u, i, u, 0)] = dh_dist_od

        logger.debug(
            'Number of compatible trips (without considering DH): {}'.format(
                len(compat_no_dh)))

        compat_arcs = list()
        for (u, i, v, j) in compat_no_dh:
            # Where does this trip end?
            src_loc = (trips_df.loc[(u, i), 'end_lat'],
                       trips_df.loc[(u, i), 'end_lon'])
            # Where does next trip start?
            dest_loc = (trips_df.loc[(v, j), 'start_lat'],
                        trips_df.loc[(v, j), 'start_lon'])

            dh_params = dh_by_coords[src_loc, dest_loc]
            dh_dist_od = dh_params['distance']
            dh_time_od = dh_params['duration']

            dh_dist[u, i, v, j] = dh_dist_od
            dh_tdelta = datetime.timedelta(minutes=dh_time_od)

            if u == v:
                # Assume always compatible within original block
                compat_arcs.append((u, i, v, j))

            else:
                # Determine whether arcs are really compatible with DH time
                time_diff = trips_df.loc[(v, j), 'start_time'] - dh_tdelta - \
                            trips_df.loc[(u, i), 'end_time']
                layover_td = datetime.timedelta(minutes=min_layover)
                max_layover_td = datetime.timedelta(minutes=max_layover)
                if max_layover_td >= time_diff >= layover_td:
                    compat_arcs.append((u, i, v, j))

        return compat_arcs, dh_dist

    def brp_wrapper(
            self, n_runs_heur=50, simple_case=False, bu_kwh=None):
        """
        Wrapper function for revising blocks using the BEB-BRP heuristic.

        :param n_runs_heur: Number of times to run heuristic with random
            trip selection
        :param simple_case: set to True if we are applying this to the
            simple case study for sensitivity analysis
        :param bu_kwh: battery capacity of all backup buses
        """
        if not bu_kwh:
            bu_kwh = max(self.chg_lims.values())

        depot_vt = (0, 0)
        max_trip_idx = {v: max(t for (v2, t) in self.veh_trip_pairs if v2 == v)
                        for v in self.infeas_vehs}
        bu_vts = [(v, t) for (v, t) in self.veh_trip_pairs
                  if v in self.infeas_vehs and t not in (0, max_trip_idx[v])]
        logger.info('Number of trips in infeasible blocks: {}'.format(
            len(bu_vts)))

        # Which pairs of trips are compatible?
        if simple_case:
            dh_fname = 'data/simple_case_deadhead.pickle'
        else:
            dh_fname = None
        compat_arcs, dh_dist = self.get_compatible_trips(
            bu_vts=bu_vts, dh_distance_fname=dh_fname, min_layover=15)

        # To get feasible arcs from compatible arcs, add depot arcs
        from_depot_arcs_z = [
            (*depot_vt, v, j) for (v, j) in bu_vts]
        to_depot_arcs_z = [
            (v, j, *depot_vt) for (v, j) in bu_vts]
        feas_arcs = compat_arcs + from_depot_arcs_z + to_depot_arcs_z
        logger.debug('Number of feasible arcs: {}'.format(len(feas_arcs)))

        heur_start = time.time()
        # TODO: don't hard-code energy rate
        best_blocks = self.run_brp_heuristic(
            infeas_vts=bu_vts, dh_dist=dh_dist, bu_kwh=bu_kwh,
            a_set=feas_arcs, method='fifo', energy_rate=3
        )

        #self.run_brp_heuristic(bu_vts, dh_dist, bu_kwh, feas_arcs, 'fifo')
        blocks_min = len(best_blocks)
        logger.debug('Number of backup buses (FIFO): {}'.format(
            len(best_blocks)))

        heur_bu_blocks = self.run_brp_heuristic(
            infeas_vts=bu_vts, dh_dist=dh_dist, bu_kwh=bu_kwh,
            a_set=feas_arcs, method='lifo', energy_rate=3
        )
        logger.debug('Number of backup buses (LIFO): {}'.format(
            len(heur_bu_blocks)))
        if len(heur_bu_blocks) < len(best_blocks):
            best_blocks = heur_bu_blocks
            blocks_min = len(heur_bu_blocks)

        blocks_tot = 0
        for heur_idx in range(n_runs_heur):
            heur_blocks = self.run_brp_heuristic(
                infeas_vts=bu_vts, dh_dist=dh_dist, bu_kwh=bu_kwh,
                a_set=feas_arcs, method='semirandom', energy_rate=3
            )
            n_bbs = len(heur_blocks)
            if n_bbs < blocks_min:
                blocks_min = n_bbs
                best_blocks = heur_blocks
            blocks_tot += n_bbs
            logger.debug('Run {}: {} backup blocks'.format(
                heur_idx, len(heur_blocks)))
        logger.info('Time for heuristic runs: {:.2f} seconds'.format(
            time.time() - heur_start))
        if n_runs_heur:
            logger.info(
                'Average number of backups: {:.2f}'.format(blocks_tot/n_runs_heur))
        logger.info(
            'Minimum number of backups: {}\n'.format(blocks_min))
        logger.debug(
            'Blocks chosen by heuristic: \n{}'.format(best_blocks))

        # Set number of backups from heuristic
        self.n_backups = blocks_min

    def set_conflict_sets(self):
        """
        Identify conflict trips for each possible charger visit.
        """
        # We only need to care about conflict sets when charging nonzero
        # charging time is feasible.
        feas_vts = [
            (v, t, s) for (v, t) in self.charging_vts for s in self.chg_sites
            if self.chg_time_avail[v, t, s] > 0]
        self.conflict_sets = {k: list() for k in feas_vts}
        chg_st = {(v, t, s): self.trip_end_times[v, t]
                  + self.trip_end_chg_times[v, t, s] for (v, t, s) in feas_vts}
        cft_time_st = time.time()
        for (v, t, s) in feas_vts:
            end_1 = chg_st[v, t, s] + self.chg_time_avail[v, t, s]
            # TODO: improve efficiency by sorting by trip start/end time
            for (v2, t2) in self.charging_vts:
                if self.chg_time_avail[v2, t2, s] > 0 and v2 != v:
                    if chg_st[v, t, s] <= chg_st[v2, t2, s] <= end_1:
                        self.conflict_sets[v, t, s].append((v2, t2))

        cft_set_time = time.time() - cft_time_st
        logger.debug('Time to set conflict sets: {:.3f} seconds'.format(
            cft_set_time))

        cflt_set_sizes = [
            len(self.conflict_sets[k]) for k in self.conflict_sets]
        if cflt_set_sizes:
            logger.info(
                'Mean Conflict set size: {:.2f}'.format(np.mean(cflt_set_sizes)))
            logger.debug(
                '{:.3f}% of conflict sets are nonempty.'.format(
                    100 * np.count_nonzero(cflt_set_sizes) / len(cflt_set_sizes)))

    def build_and_solve(self, alpha, opt_gap=None, add_cuts=False):
        """
        Solve BEB charger facility location problem.

        :param alpha: Objective function parameter alpha
        :param opt_gap: Relative optimality gap for Gurobi MIP solver
        """
        self.alpha = alpha
        # Pre-processing: which charging sites are feasible?
        self.s_vt = {(v, t): [s for s in self.chg_sites
                         if self.chg_time_avail[v, t, s] > 0]
                     for (v, t) in self.charging_vts}
        vts_tups = {(v, t, s) for (v, t) in self.charging_vts
                   for s in self.s_vt[v, t]}
        logger.debug('|V||T||S|: {}'.format(
            len(self.charging_vts) * len(self.chg_sites)))
        logger.debug('|V, T, S|: {}'.format(
            len(vts_tups)))

        # Build Pyomo model
        # First, define sets
        m = ConcreteModel()
        m.chg_sites = Set(initialize=self.chg_sites)
        m.vehicles = Set(initialize=self.charging_vehs)
        m.vt_pairs = Set(dimen=2, initialize=self.charging_vts)
        m.vts_tups = Set(dimen=3, initialize=vts_tups)

        # Variables
        m.battery_charge = Var(m.vt_pairs, within=NonNegativeReals)
        m.site_binary = Var(m.chg_sites, within=Binary)
        m.num_chargers = Var(m.chg_sites, within=NonNegativeIntegers)
        m.chg_binary = Var(m.vts_tups, within=Binary)
        m.chg_time = Var(m.vts_tups, within=NonNegativeReals)

        def chg_vars_relation(model, v, t, s):
            return model.chg_time[v, t, s] <= self.chg_time_avail[
                v, t, s] * model.chg_binary[v, t, s]
        m.chg_vars_constr = Constraint(m.vts_tups, rule=chg_vars_relation)

        def single_site_charging(model, v, t):
            if self.s_vt[v, t]:
                return sum(model.chg_binary[v, t, s] for s in self.s_vt[v, t]) <= 1
            else:
                return Constraint.Feasible
        m.single_charge_constr = Constraint(
            m.vt_pairs, rule=single_site_charging)

        def n_sites_relation(model, s):
            return model.num_chargers[s] <= self.max_chargers[
                s] * model.site_binary[s]
        m.n_sites_constr = Constraint(m.chg_sites, rule=n_sites_relation)

        def x_y_relation(model, v, t, s):
            return model.chg_binary[v, t, s] <= model.num_chargers[s]
        m.x_y_constr = Constraint(m.vts_tups, rule=x_y_relation)

        def init_charge(model, v, t):
            if t == 0:
                return model.battery_charge[v, t] == self.chg_lims[v]
            else:
                return Constraint.Skip
        m.init_chg_constr = Constraint(m.vt_pairs, rule=init_charge)

        def charge_tracking(model, v, t):
            # Don't apply to last trip
            if (v, t + 1) not in self.charging_vts:
                return Constraint.Skip

            return model.battery_charge[v, t + 1] == model.battery_charge[v, t] \
                   + sum(self.chg_rates[s] * model.chg_time[v, t, s]
                         - self.energy_rates[v, t] * (
                                 self.trip_end_chg_dists[v, t, s]
                                 + self.trip_start_chg_dists[v, t + 1, s])
                         * model.chg_binary[v, t, s] for s in self.s_vt[v, t]) \
                   - self.energy_rates[v, t] * self.trip_dists[v, t] \
                   - self.energy_rates[v, t] * self.inter_trip_dists[v, t] * (
                           1 - sum(
                       model.chg_binary[v, t, s] for s in self.s_vt[v, t]))
        m.chg_track_constr = Constraint(m.vt_pairs, rule=charge_tracking)

        def charge_min(model, v, t):
            return model.battery_charge[v, t] - self.energy_rates[v, t] * \
                   self.trip_dists[v, t] - sum(
                self.energy_rates[v, t] * self.trip_end_chg_dists[v, t, s] *
                model.chg_binary[v, t, s] for s in self.s_vt[v, t]) \
                   >= 0
        m.charge_min_constr = Constraint(m.vt_pairs, rule=charge_min)

        def charge_max(model, v, t):
            return model.battery_charge[v, t] - self.energy_rates[v, t] * \
                   self.trip_dists[v, t] + sum(
                self.chg_rates[s] * model.chg_time[v, t, s] -
                self.energy_rates[v, t] * self.trip_end_chg_dists[v, t, s] *
                model.chg_binary[v, t, s] for s in self.s_vt[v, t]) \
                   <= self.chg_lims[v]
        m.charge_max_constr = Constraint(m.vt_pairs, rule=charge_max)

        # Cuts because we know vehicles must charge before running out
        # of battery, so accordingly we must build stations they can
        # reach.
        def charge_site_cut(model, v):
            reachable_sites = set(
                [s for t in self.trips_per_veh[v] for s in self.s_vt[v, t]
                 if t <= self.last_feas_trips[v]])
            return sum(model.site_binary[s] for s in reachable_sites) >= 1
        if add_cuts:
            m.site_cut_constr = Constraint(m.vehicles, rule=charge_site_cut)

        # Objective function
        start_times = {(v, t + 1, s): self.trip_start_chg_times[v, t + 1, s]
                       if (v, t + 1, s) in self.trip_start_chg_times else 0
                       for (v, t) in self.charging_vts
                       for s in self.s_vt[v, t]}

        def min_cost(model):
            return sum(
                self.site_costs[s] * model.site_binary[s]
                + self.charger_costs[s] * model.num_chargers[s]
                for s in model.chg_sites) + alpha * sum(
                (self.trip_end_chg_times[v, t, s]
                 + start_times[v, t + 1, s] - self.inter_trip_times[v, t])
                * model.chg_binary[v, t, s] for (v, t, s) in model.vts_tups)
        m.obj = Objective(rule=min_cost)

        # Create solver object and solve
        solver = SolverFactory('gurobi')
        if opt_gap is not None:
            solver.options['MIPgap'] = opt_gap

        # solver.options['presolve'] = 2
        # solver.options['NodeMethod'] = 0
        # solver.options['CoverCuts'] = 2
        solver.options['MIPFocus'] = 3
        solver.options['Method'] = 1
        start_time = time.time()
        results = solver.solve(m, tee=False)
        init_time = time.time() - start_time
        logger.info('Time for initial model solve: {:.2f} seconds'.format(
            init_time))

        infeas_flag = True
        constr_ixs = dict()
        itr = 1
        while infeas_flag and itr <= 100:
            # Check if capacity constraints are violated
            vts_chg = [(v, t, s) for (v, t, s) in m.vts_tups
                       if value(m.chg_binary[v, t, s]) > 0.99]
            new_infeas = False
            for (v, t, s) in vts_chg:
                conflict_vt = list()
                for (v2, t2) in self.conflict_sets[v, t, s]:
                    if value(m.chg_binary[v2, t2, s] > 0.99):
                        conflict_vt.append((v2, t2))
                if len(conflict_vt) + 1 > np.ceil(value(m.num_chargers[s])):
                    # Constraint is violated
                    new_infeas = True
                    if (v, t, s) in constr_ixs:
                        logger.info(
                            'Already included constraint for {}, {}, {}'.format(
                                v, t, s))
                    else:
                        constr_ixs[v, t, s] = conflict_vt

            if not new_infeas:
                infeas_flag = False

            else:
                logger.info(
                    'Incumbent model has {} queue prevention constraints'.format(
                        len(constr_ixs)))

                if itr > 1:
                    m.del_component(m.capacity_constr)
                    try:
                        m.del_component(m.capacity_constr_index)
                    except AttributeError:
                        # Suddenly starting getting an error here sometimes,
                        # probably due to a pyomo update?
                        logger.info('no index to delete')
                        pass

                def site_capacity(model, v, t, s):
                    return model.chg_binary[v, t, s] + sum(
                        model.chg_binary[v2, t2, s] for (v2, t2) in self.conflict_sets[
                            v, t, s]) <= model.num_chargers[s]
                m.capacity_constr = Constraint(
                    list(constr_ixs.keys()), rule=site_capacity)

                itr_start_time = time.time()
                results = solver.solve(m, tee=False, warmstart=True)
                self.soln_time = time.time() - itr_start_time
                logger.info('Time for model solve {}: {:.2f} seconds'.format(
                    itr, self.soln_time))
                itr += 1

        self.soln_time = time.time() - start_time
        logger.info('Time for complete model solve: {:.2f} seconds'.format(
            self.soln_time))
        self.process_solver_output(results, m)
        self.process_results()

    def solve(self, alpha, opt_gap=None, simple_case=False, bu_kwh=None,
              n_runs_heur=50):
        """Wrapper function to run BEB-BRP and BEB-OCL successively."""
        self.check_charging_needs()
        self.check_feasibility()
        if self.infeas_vehs:
            self.brp_wrapper(simple_case=simple_case, bu_kwh=bu_kwh,
                             n_runs_heur=n_runs_heur)
        self.charging_vehs = [v for v in self.charging_vehs
                              if v not in self.infeas_vehs]
        self.charging_vts = [(v, t) for (v, t) in self.charging_vts
                             if v in self.charging_vehs]
        self.set_max_chg_time()
        self.set_conflict_sets()
        self.build_and_solve(alpha, opt_gap)

    def process_solver_output(self, results, solved_model):
        """
        Process pyomo solver output to ensure result is optimal.

        :param results: pyomo results object
        :param solved_model: solved pyomo ConcreteModel object
        """
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition
                == TerminationCondition.optimal):
            self.solver_status = 'optimal'
            self.model = solved_model
        # Do something when the solution in optimal and feasible
        elif (results.solver.termination_condition
              == TerminationCondition.infeasible):
            self.solver_status = 'infeasible'
            raise ValueError('Solver found model to be infeasible.')
        # Do something when model in infeasible
        else:
            # Something else is wrong
            self.solver_status = str(results.solver.status)
            self.model = solved_model
            raise Warning('Solver returned unrecognized status: {}'.format(
                results.solver.status))

    def process_results(self):
        """Process results of solved model into more useful outputs."""
        # Get objective function value
        self.opt_cost = value(self.model.obj)

        # Which stations are built?
        self.opt_stations = [s for s in self.chg_sites
                             if np.abs(value(self.model.site_binary[s]) - 1)
                             < 1e-3]
        self.num_chargers = {s: int(value(self.model.num_chargers[s]))
                             for s in self.chg_sites}
        self.capital_costs = sum(
            self.site_costs[s] * np.round(value(self.model.site_binary[s]))
            + self.charger_costs[s] * np.round(value(self.model.num_chargers[s]))
            for s in self.chg_sites)

        # When and where does charging happen?
        self.chg_schedule = {(v, t, s): value(self.model.chg_time[v, t, s])
                             for (v, t) in self.charging_vts
                             for s in self.s_vt[v, t]}

        # Calculate total time spent charging
        self.opt_total_chg_time = sum(
            value(self.model.chg_time[v, t, s]) for (v, t) in self.charging_vts
            for s in self.s_vt[v, t])

        # Calculate number of charges
        self.opt_charges = np.round(
            sum(value(self.model.chg_binary[v, t, s])
                for (v, t) in self.charging_vts for s in self.s_vt[v, t]),
            0
        )
        vts_chg_tups = [
            (v, t, s) for (v, t) in self.charging_vts
            for s in self.s_vt[v, t] if value(self.model.chg_binary[v, t, s])
            > 0.99]
        # Calculate charge time as a fraction of total time available
        self.chg_ratios = {
            k: self.chg_schedule[k]/self.chg_time_avail[k]
            for k in vts_chg_tups}

        start_times = {
            (v, t + 1, s): self.trip_start_chg_times[v, t + 1, s]
            if (v, t + 1, s) in self.trip_start_chg_times else 0
            for (v, t) in self.charging_vts for s in self.s_vt[v, t]}
        self.opt_deadhead_time = sum(
                (self.trip_end_chg_times[v, t, s] + start_times[v, t+1, s]
                 - self.inter_trip_times[v, t])
                * value(self.model.chg_binary[v, t, s])
                for (v, t) in self.charging_vts for s in self.s_vt[v, t])

        # Calculate charger usage
        chg_intervals = {s: list() for s in self.chg_sites}
        chg_idxs = [(v, t, s) for (v, t) in self.charging_vts
                    for s in self.s_vt[v, t]
                    if value(self.model.chg_binary[v, t, s]) == 1]
        for (v, t, s) in chg_idxs:
            station_arr = self.trip_end_times[v, t] + self.trip_end_chg_times[
                v, t, s]
            station_dept = station_arr + value(self.model.chg_time[v, t, s])
            chg_intervals[s].append((station_arr, station_dept))
        self.chg_intervals = chg_intervals

        self.chgs_per_veh = {v: sum(value(
            self.model.chg_binary[v, t, s])
            for (v2, t) in self.charging_vts if v2 == v
            for s in self.s_vt[v, t]) for v in self.vehicles}

    def log_results(self):
        """Report summary results using logging module."""
        logger.info('Total number of backup buses used: {}'.format(self.n_backups))
        logger.info('Optimal objective function value: {:.2f}'.format(self.opt_cost))
        logger.info('Optimal stations: {}'.format(self.opt_stations))
        logger.info('Number of chargers: {}'.format(self.num_chargers))
        logger.info('Optimal number of charger visits: {}'.format(self.opt_charges))
        logger.info('Optimal total charging time: {:.2f} minutes'.format(
            self.opt_total_chg_time))
        logger.info('Average time per charge: {:.2f} minutes'.format(
            self.opt_total_chg_time / self.opt_charges))

    def plot_chg_ratios(self, save_fname=None):
        """
        Plot histogram of charging time as a fraction of feasible max.
        :param save_fname: (optional) filename to save plot in. If
            `None`, plot will be shown but not saved.
        """
        plt.figure()
        rvals = list(self.chg_ratios.values())
        plt.hist(rvals)
        plt.xlabel('Charging Ratio')
        plt.ylabel('Count')
        plt.axvline(np.mean(rvals), color='k', linestyle='dashed', linewidth=1)
        plt.tight_layout()
        if save_fname:
            plt.savefig(save_fname[:-4] + '_full' + save_fname[-4:], dpi=350)
        else:
            plt.show()

    def plot_chargers(self, save_fname=None):
        """
        Create a plot of charger utilization.

        This method can be called after a model is solved and its results
        are processed in order to plot the utilization of chargers at all
        locations. It is largely copied in a function of the same name in
        :py:mod:`beb_vis`.

        :param save_fname: (optional) filename to save plot in. If `None`,
            plot will be shown but not saved.
        """
        plt.rcParams.update({'font.size': 16})
        # Get start and end times for plotting
        x_lb = min(self.trip_start_times.values())
        x_ub = max([self.trip_end_times[v, t] for (v, t) in self.charging_vts])
        used_sites = [s for s in self.chg_sites
                      if any(value(self.model.chg_binary[v, t, ss]) == 1
                             for (v, t) in self.charging_vts
                             for ss in self.s_vt[v, t] if ss == s)]
        n_sites = len(used_sites)

        fig, axs = plt.subplots(nrows=n_sites, ncols=1, sharex=True,
                                sharey=True, figsize=(8, 3*n_sites))

        for i, s in enumerate(used_sites):
            start_times = sorted([j[0] for j in self.chg_intervals[s]])
            end_times = sorted([j[1] for j in self.chg_intervals[s]])
            x = [x_lb] + sorted(start_times + end_times) + [x_ub]
            x_dt = [self.zero_time + datetime.timedelta(minutes=m) for m in x]
            y = [0]*len(x)
            start_idx = 0
            end_idx = 0
            while start_idx < len(start_times) and end_idx < len(end_times):
                y_idx = start_idx + end_idx + 1
                start_val = start_times[start_idx]
                end_val = end_times[end_idx]
                if start_val < end_val:
                    y[y_idx] = y[y_idx-1] + 1
                    start_idx += 1
                elif end_val < start_val:
                    y[y_idx] = y[y_idx-1] - 1
                    end_idx += 1
                else:
                    # New charge begins as soon as another ends
                    y[y_idx] = y[y_idx-1]
                    y[y_idx+1] = y[y_idx]
                    start_idx += 1
                    end_idx += 1

            axs[i].set_title('Charging Site: {}'.format(s))
            axs[i].step(x_dt, y, where='post')
            axs[i].xaxis.set_major_locator(HourLocator(interval=4))
            axs[i].xaxis.set_major_formatter(DateFormatter('%H:%M'))
            axs[i].yaxis.set_major_locator(ticker.AutoLocator())

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                        right=False)
        plt.ylabel('Number of Buses Charging')
        plt.xlabel('Time')
        plt.tight_layout()
        if save_fname:
            plt.savefig(save_fname, dpi=350)
        plt.show()

    def plot_conflict_sets(self, save_fname=None):
        """
        Plot histogram showing size of all conflict sets.
        :param save_fname: filename to save plot in. If `None`,
            plot will be shown but not saved.
        """
        cflt_set_sizes = [
            len(self.conflict_sets[k]) for k in self.conflict_sets]

        fig, ax = plt.subplots()
        bins = [-0.5 + i for i in range(max(cflt_set_sizes) + 2)]
        plt.hist(cflt_set_sizes, bins=bins, rwidth=0.8)
        # plt.title('Histogram of Conflict Set Sizes')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel('Size of Conflict Set')
        plt.ylabel('Count')
        plt.tight_layout()
        if save_fname:
            plt.savefig(save_fname, dpi=350)
        else:
            plt.show()

    def pickle(self, fname: str):
        """
        Save the model object with pickle. This solved model can then be
        analyzed further with the simulation code without having to
        solve it again from scratch.

        :param fname: filename to save to
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def to_df(self):
        """
        Return all details of solved model in a DataFrame.

        :param fname: csv filename
        :return: results DataFrame
        """
        trips_df = self.trips_df.set_index(['block_id', 'trip_idx'])
        if self.solver_status != 'optimal':
            raise ValueError('Cannot output solution when solver status was '
                             'not optimal.')

        v_list = list()
        t_list = list()
        t_ids = list()
        start_times = list()
        end_times = list()
        opt_sites = list()
        opt_chgs = list()
        opt_soc = list()
        opt_dh1 = list()
        opt_dh2 = list()
        opt_dh3 = list()
        for i, (v, t) in enumerate(self.charging_vts):
            v_list.append(v)
            t_list.append(t)
            start_times.append(self.trip_start_times[v, t])
            end_times.append(self.trip_end_times[v, t])
            try:
                # TODO: make sure this always gives consistent behavior. We
                #   have indexing conventions that can be a little
                #   confusing. Trips are indexed starting with zero (leaving
                #   the depot) in the optimization, so trip 1 in
                #   self.charging_vts corresponds to trip 0 in trips_df
                t_ids.append(trips_df.loc[v, t-1]['trip_id'])
            except KeyError:
                # KeyError happens when t == 0 (departure from depot)
                # or t == max for this block (return to depot)
                if t == 0:
                    # Use ID 0 as placeholder for departure from depot
                    t_ids.append(0)
                elif t == max(tt for (vv, tt) in self.veh_trip_pairs
                              if vv == v):
                    # Use ID 100 as placeholder for return to depot
                    t_ids.append(100)
                else:
                    print(trips_df.head())
                    raise ValueError(
                        'Unrecognized trip for block {}, trip index {}'.format(
                            v, t))

            chg_by_site = {
                s: self.chg_schedule[v, t, s] for s in self.opt_stations
                if s in self.s_vt[v, t]}

            if not chg_by_site:
                chg_by_site = {None: 0}

            if max(chg_by_site.values()) > 1e-6:
                site_i = max(chg_by_site, key=chg_by_site.get)
                chg_i = self.chg_schedule[v, t, site_i]
                dh1_i = self.trip_end_chg_times[v, t, site_i]
                try:
                    dh2_i = self.trip_start_chg_times[v, t+1, site_i]
                except KeyError:
                    dh2_i = 0
                dh3_i = self.inter_trip_dists[v, t]

            else:
                site_i = np.nan
                chg_i = np.nan
                dh1_i = np.nan
                dh2_i = np.nan
                dh3_i = np.nan

            opt_sites.append(site_i)
            opt_chgs.append(chg_i)
            opt_soc.append(value(self.model.battery_charge[v, t]))
            opt_dh1.append(dh1_i)
            opt_dh2.append(dh2_i)
            opt_dh3.append(dh3_i)

        out_dict = {
            'block_id': v_list,
            'trip_id': t_ids,
            'trip_idx': t_list,
            'start_time': start_times,
            'end_time': end_times,
            'soc': opt_soc,
            'chg_site': opt_sites,
            'chg_time': opt_chgs,
            'dh1': opt_dh1,
            'dh2': opt_dh2,
            'dh3': opt_dh3}
        out_df = pd.DataFrame(out_dict)
        return out_df

    def to_csv(self, fname):
        """
        Save all details of solved model to csv.

        :param fname: csv filename
        """
        out_df = self.to_df()
        out_df.to_csv(fname, index=False)

    def summary_to_df(self, *args, **kwargs):
        """
        Return summary values of a solved model as a DataFrame.

        **NOTE:**
        `*args` and `**kwargs` provide two different ways to add details
        to the results in the form (name, value). For example, if we
        wanted to document that rho=100, we could use keyword args
        {'rho': 100} or non-keyword arguments ['rho', 100].
        """
        if self.solver_status != 'optimal':
            raise ValueError('Cannot output solution when solver status was '
                             'not optimal.')

        if len(args) % 2 != 0:
            raise ValueError(
                'Only an even number of non-keyword arguments may be '
                'given to this function. Each pair of arguments should '
                'be of the form (param name, param value).')

        param_dict = dict()
        arg_idx = range(int(len(args) / 2))
        for a in arg_idx:
            name = args[a*2]
            val = args[a*2 + 1]
            param_dict[name] = val

        for kw in kwargs:
            param_dict[kw] = kwargs[kw]

        out_dict = {
            'obj_val': self.opt_cost,
            'n_backups': self.n_backups,
            'num_stations': len(self.opt_stations),
            'num_chargers': sum(self.num_chargers.values()),
            'num_charges': self.opt_charges,
            'charge_time': self.opt_total_chg_time,
            'deadhead_time': self.opt_deadhead_time,
            'operations_cost': self.alpha * self.opt_deadhead_time,
            'recov_time_lost': self.opt_deadhead_time + self.opt_total_chg_time,
            'capital_cost': self.capital_costs}
        out_dict.update(param_dict)

        # Add number of chargers built at each site
        for s in self.chg_sites:
            col_str = 'N at {}'.format(s)
            out_dict[col_str] = self.num_chargers[s]
        out_df = pd.DataFrame(out_dict, index=[0])
        out_df['stations'] = None
        out_df.at[0, 'stations'] = self.opt_stations
        return out_df

    def summary_to_csv(self, fname, *args, **kwargs):
        """
        Save summary values of a solved model in CSV format.

        :param fname: filename to save to

        **NOTE:**
        `*args` and `**kwargs` provide two different ways to add details
        to the results in the form (name, value). For example, if we
        wanted to document that rho=100, we could use keyword args
        {'rho': 100} or non-keyword arguments ['rho', 100].
        """
        out_df = self.summary_to_df(*args, **kwargs)
        out_df.to_csv(fname, index=False)

