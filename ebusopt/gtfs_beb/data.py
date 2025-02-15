import pandas as pd
import googlemaps
import pickle
import logging
import time
import numpy as np
from pathlib import Path
from collections.abc import Iterable
from datetime import datetime, timedelta
from openrouteservice import client
from dotenv import dotenv_values, find_dotenv

logger = logging.getLogger('gtfs_data')


class GTFSError(Exception):
    pass


class GTFSData:
    """
    Class for reading and processing GTFS data.
    """
    def __init__(
            self, calendar_file, calendar_dates_file, trips_file, shapes_file,
            routes_file, stop_times_file):
        """
        Constructor for GTFSData class.

        The GTFSData class is responsible for storing and processing
        static GTFS files so that they can be used by the optimization
        methods in :py:mod:`beb_model`.

        :param calendar_file:
        :param calendar_dates_file:
        :param trips_file:
        :param shapes_file:
        :param routes_file:
        :param stop_times_file:
        """

        self.trips_df = self._load_table(
            filename=trips_file,
            required_cols=[
                'trip_id', 'route_id', 'service_id', 'block_id', 'shape_id'
            ],
            dtype={
                'trip_id': str,
                'route_id': str,
                'service_id': str,
                'block_id': str,
                'shape_id': str
            }
        )
        self.routes_df = self._load_table(
            filename=routes_file,
            required_cols=['route_id', 'route_short_name', 'route_type'],
            optional_cols=['route_desc'],
            dtype={
                'route_id': str,
                'route_short_name': str,
                'route_type': int
            }
        )
        self.routes_df.set_index('route_id', inplace=True)
        self._filter_by_trip_type()

        # At least one of calendar.txt and calendar_dates.txt is
        # required, but it's okay to have just one of them.
        try:
            self.calendar_df = self._load_table(
                filename=calendar_file,
                required_cols=[
                    'service_id', 'monday', 'tuesday', 'wednesday',
                    'thursday', 'friday', 'saturday', 'sunday', 'start_date',
                    'end_date'
                ],
                dtype={
                    'service_id': str
                }
            )
            self.calendar_df['start_date'] = pd.to_datetime(
                self.calendar_df['start_date'].astype(str))
            self.calendar_df['end_date'] = pd.to_datetime(
                self.calendar_df['end_date'].astype(str))
        except FileNotFoundError:
            self.calendar_df = None

        try:
            self.calendar_dates_df = self._load_table(
                filename=calendar_dates_file,
                required_cols=[
                    'service_id', 'date', 'exception_type'
                ],
                dtype={
                    'service_id': str
                }
            )
            self.calendar_dates_df['date'] = pd.to_datetime(
                self.calendar_dates_df['date'].astype(str))
        except FileNotFoundError:
            self.calendar_dates_df = None

        if self.calendar_df is None and self.calendar_dates_df is None:
            raise GTFSError(
                'At least one of calendar.txt and calendar_dates.txt must '
                'be provided.'
            )

        self.shapes_df = self._load_table(
            shapes_file,
            required_cols=[
                'shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'
            ],
            dtype={
                'shape_id': str
            }
        )
        self.stop_times_df = self._load_table(
            stop_times_file,
            required_cols=['trip_id', 'arrival_time'],
            dtype={'trip_id': str}
        )
        self.shapes_summary_df = None

    @classmethod
    def from_dir(cls, dir_name: str | Path):
        trips_file = '{}/trips.txt'.format(dir_name)
        calendar_file = '{}/calendar.txt'.format(dir_name)
        calendar_dates_file = '{}/calendar_dates.txt'.format(dir_name)
        shapes_file = '{}/shapes.txt'.format(dir_name)
        routes_file = '{}/routes.txt'.format(dir_name)
        stop_times_file = '{}/stop_times.txt'.format(dir_name)

        return cls(
            calendar_file=calendar_file,
            calendar_dates_file=calendar_dates_file,
            trips_file=trips_file,
            shapes_file=shapes_file,
            routes_file=routes_file,
            stop_times_file=stop_times_file
        )

    @staticmethod
    def from_pickle(fname):
        """
        Unpickle a GTFSData object.

        :param fname: filename of pickled GTFSData object
        :return: unpickled object
        """
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def _load_table(
            filename, required_cols, optional_cols=None, index_col=None,
            dtype=None
    ):
        """
        Confirm that the given GTFS table contains all required columns.

        :param filename: path to file to check
        :param required_cols: columns that must be included in the table
            If they are missing, raise a ValueError.
        :param optional_cols: columns that should be included in the
            returned DataFrame if they are present. No error will be
            raised if they are missing.
        :param dtype: dict giving column data types, optional
        :return: DataFrame of loaded data, or raise an appropriate error
        """
        df = pd.read_csv(filename, index_col=index_col, dtype=dtype)
        if all(c in df.columns for c in required_cols):
            if optional_cols is not None:
                if not isinstance(optional_cols, list):
                    raise TypeError(
                        'Expected type list, but received type {}'.format(
                            type(optional_cols)
                        )
                    )

                present_optional_cols = [
                    c for c in optional_cols if c in df.columns]

                return df[required_cols + present_optional_cols]

            else:
                return df[required_cols]

        else:
            missing_cols = [c for c in required_cols if c not in df.columns]
            raise GTFSError(
                'The provided file {} is missing the following columns that '
                'are required for our analysis: {}'.format(
                    filename, missing_cols))


    @staticmethod
    def filter_df(df, col, value):
        """
        Filter down the given DataFrame, returning all rows where the
        column "col" has the given value "value"

        :param df: a DataFrame
        :param col: column of DataFrame to filter on
        :param value: desired value in column, may be a single value or
            an iterable
        :return: Filtered DataFrame
        """
        # If value is iterable, check if column value is in it
        if isinstance(value, Iterable):
            return df[df[col].isin(value)]
        # Otherwise, just check equality
        else:
            return df[df[col] == value]

    def pickle(self, fname):
        """
        Pickle this object.

        :param fname: filename to save to
        :return: nothing, just saves file
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def _filter_by_trip_type(self):
        """
        Remove any non-bus trips from our data.
        """
        # Which routes are not bus routes?
        bad_rts = self.routes_df[
            self.routes_df['route_type'] != 3].index.tolist()
        # Exclude any trips on those routes.
        self.trips_df = self.trips_df[
            ~self.trips_df['route_id'].isin(bad_rts)]

    def calculate_shape_dists(self):
        """
        Use the shape points provided in shape_df to calculate the
        total length of all shapes included in the table.
        """
        gb = self.shapes_df.groupby('shape_id')
        shapes_list = list()
        dists_list = list()
        for shape_id, shape_grp in gb:
            shape_grp = shape_grp.sort_values(
                by='shape_pt_sequence').reset_index()
            # Add columns that give coordinates of last point
            shape_grp[['prev_lat', 'prev_lon']] = shape_grp[
                ['shape_pt_lat', 'shape_pt_lon']].shift(1)
            # First point in the shape doesn't have a predecessor. Just
            # set it to itself so the distance is zero.
            shape_grp.loc[0, 'prev_lat'] = shape_grp.loc[0, 'shape_pt_lat']
            shape_grp.loc[0, 'prev_lon'] = shape_grp.loc[0, 'shape_pt_lon']
            # Calculate distance between points
            shape_grp['seq_dist'] = haversine_np(
                shape_grp['shape_pt_lon'], shape_grp['shape_pt_lat'],
                shape_grp['prev_lon'], shape_grp['prev_lat'])
            # Add total distance to list
            shapes_list.append(shape_id)
            dists_list.append(shape_grp['seq_dist'].sum())

        # Build dataframe of results
        manual_dists = pd.DataFrame(
            {'shape_id': shapes_list, 'manual_dist': dists_list}
        ).set_index('shape_id')
        return manual_dists

    def summarize_shapes(self, shape_ids):
        """
        Create a summary table of key shape details.

        This method aggregates the data in shape_df to get some details
        needed by our models: the lat/lon coordinates of the start and
        end location of the trip and its total distance. This is stored
        as self.shapes_summary_df so we can efficiently access this
        summary info when needed.

        :param shape_ids: list of shape_id values that should be
            summarized. saves computation time if we don't need to
            process all shape_ids in the table.
        """
        if self.shapes_summary_df is None:
            shapes_to_add = shape_ids
        else:
            shapes_to_add = [
                s for s in shape_ids
                if s not in self.shapes_summary_df.index.unique()
            ]

        if not shapes_to_add:
            return

        # Filter down to just the shapes we need to add
        shapes_df = self.shapes_df[self.shapes_df['shape_id'].isin(
            shapes_to_add)]
        shapes_multi = shapes_df.set_index(
            ['shape_id', 'shape_pt_sequence'])

        shapes_summary = pd.DataFrame()
        shapes_summary[['start_lat', 'start_lon']] = shapes_multi.groupby(
            'shape_id').apply(lambda x: x.sort_index().iloc[0])[
            ['shape_pt_lat', 'shape_pt_lon']]
        shapes_summary[['end_lat', 'end_lon']] = \
            shapes_multi.groupby('shape_id').apply(
                lambda x: x.sort_index().iloc[-1])[
                ['shape_pt_lat', 'shape_pt_lon']]
        # Calculate distances of all shapes (in miles)
        shape_dists_df = self.calculate_shape_dists()
        # Add these to the summary DF
        shapes_summary['total_dist'] = shape_dists_df
        # Update the summary df
        if self.shapes_summary_df is None:
            self.shapes_summary_df = shapes_summary
        else:
            self.shapes_summary_df = pd.concat(
                [self.shapes_summary_df, shapes_summary]
            )

    def get_service_ids(self, input_date):
        """
        Get service IDs corresponding to the given input date.

        :param input_date: datetime.datetime object giving specific date
        :return: list of service_ids in service on the given date
        """
        if self.calendar_df is None:
            ids_cal = list()
        else:
            after_start = input_date >= self.calendar_df['start_date']
            before_end = input_date <= self.calendar_df['end_date']
            calendar_filt = self.calendar_df[(after_start & before_end)]

            dow = input_date.weekday()
            dow_dict = {
                0: 'monday',
                1: 'tuesday',
                2: 'wednesday',
                3: 'thursday',
                4: 'friday',
                5: 'saturday',
                6: 'sunday'
            }
            dow_col = dow_dict[dow]
            calendar_filt = calendar_filt[calendar_filt[dow_col] == 1]

            ids_cal = calendar_filt['service_id'].tolist()

        if self.calendar_dates_df is None:
            service_add = list()
            service_rmv = list()
        else:
            service_except = self.calendar_dates_df[
                self.calendar_dates_df['date'] == input_date]
            if len(service_except) > 0:
                service_add = service_except[
                    service_except['exception_type'] == 1][
                    'service_id'].tolist()
                service_rmv = service_except[
                    service_except['exception_type'] == 2][
                    'service_id'].tolist()

            else:
                service_add = list()
                service_rmv = list()

        # Combine the results
        all_ids = set(ids_cal + service_add)
        if service_add:
            len_before = len(set(ids_cal))
            logger.debug('Additions from calendar_dates: {}'.format(
                len(all_ids) - len_before))

        if service_rmv:
            len_before = len(set(all_ids))
            all_ids = set([i for i in all_ids if i not in service_rmv])
            logger.debug('Removals from calendar_dates: {}'.format(
                len(all_ids) - len_before))

        return all_ids

    def add_trip_data(self, df, ref_date):
        """
        Add relevant details from other GTFS tables to data from trips.txt.

        This function takes as input a DataFrame containing any subset
        of trips as defined in trips.txt and adds the following fields:
            - start_time
            - end_time
            - trip_idx
            - start_lat
            - start_lon
            - end_lat
            - end_lon

        Note that setting trip_idx will NOT work as expected unless
        ALL trips with a given block_id are provided in the input df.
        Otherwise, trip indexes will not be correct.

        :param df: DataFrame containing some subset of block_id values
        :return: DataFrame with fields from other tables added
        """
        if len(df) == 0:
            raise ValueError('Empty DataFrame')

        # Get stop times only for the relevant trips
        st_filt = self.filter_df(
            self.stop_times_df, 'trip_id', df['trip_id'].tolist())
        # TODO: use pd.to_timedelta() instead of to_datetime_safe()
        # Convert times from string to datetime
        try:
            st_filt.loc[:, 'arrival_time'] = to_datetime_safe(
                st_filt['arrival_time'], ref_date
            )
        except GTFSError:
            pass
        # Get start time of every trip
        start_times = st_filt.groupby('trip_id').apply(
            lambda x: x['arrival_time'].min()).sort_values().rename(
            'start_time')
        # Get end time of every trip
        end_times = st_filt.groupby('trip_id').apply(
            lambda x: x['arrival_time'].max()).sort_values().rename('end_time')
        # Merge start and end times together
        trip_times = pd.merge(
            start_times, end_times, left_index=True, right_index=True)
        # Add this data to trips DF
        trips_mrg = pd.merge(
            df, trip_times, left_on='trip_id', right_index=True)
        # Add trip indexes
        trips_mrg = trips_mrg.sort_values(by=['block_id', 'start_time'])
        trips_mrg['trip_idx'] = trips_mrg.groupby('block_id').cumcount() + 1

        # Merge in start/end coords and distance from shapes summary
        # First, we need to ensure the trips are processed in
        # shapes_summary_df to get their coords and distances.
        shape_id_list = list(trips_mrg['shape_id'].unique())
        self.summarize_shapes(shape_ids=shape_id_list)

        trips_mrg = pd.merge(
            trips_mrg, self.shapes_summary_df, left_on='shape_id',
            right_index=True)

        return trips_mrg

    @staticmethod
    def add_deadhead(trips_df):
        trips_df = trips_df.sort_values(by=['block_id', 'trip_idx'])
        block_gb = trips_df.groupby('block_id')
        dh_dfs = list()
        for block_id, block_df in block_gb:
            block_df[['dh_dest_lat', 'dh_dest_lon']] = block_df[
                ['start_lat', 'start_lon']].shift(-1)
            dh_dfs.append(block_df)
        trips_df = pd.concat(dh_dfs)

        # Gather unique OD pairs for all deadhead trips
        od_pairs = trips_df[
            ['end_lat', 'end_lon', 'dh_dest_lat', 'dh_dest_lon']]
        od_pairs = od_pairs.drop_duplicates().dropna().reset_index(drop=True)

        # Use Manhattan distance to calculate deadhead distance. Much
        # faster/easier than using a distance API and should be close.
        od_pairs['dh_dist'] = manhattan_np(
            od_pairs['end_lon'], od_pairs['end_lat'],
            od_pairs['dh_dest_lon'], od_pairs['dh_dest_lat']
        )

        # Merge DH dists into trip DF
        merged_trips = pd.merge(
            trips_df, od_pairs,
            on=['end_lat', 'end_lon', 'dh_dest_lat', 'dh_dest_lon'],
            how='left'
        ).fillna({'dh_dist': 0})

        return merged_trips

    def get_trips_from_sids(self, sids, ref_date=None, add_data=False):
        """
        Get all trips with the given service_id value(s)
        :param sids: single service_id or list of service_id values
        :param ref_date: reference date used in time columns
        :param add_data: True if all trip data (e.g. start/end coords,
            stop times, and trip distances) should be added to the
            returned DataFrame
        :return: DataFrame of trip data for matching trips
        """
        # Get only the relevant trips
        trips_filt = self.filter_df(self.trips_df, 'service_id', sids)
        # Merge in route data (name, description, and route type)
        trips_mrg = pd.merge(
            trips_filt, self.routes_df, left_on='route_id', right_index=True)

        # Only keep bus routes
        trips_mrg = self.filter_df(trips_mrg, 'route_type', 3)
        if add_data:
            # Get stop times and shape data only for the relevant trips
            trips_mrg = self.add_trip_data(trips_mrg, ref_date)

        return trips_mrg

    def get_trips_from_date(self, input_date):
        """
        Given a date, gather all needed data on all trips operating.

        :param input_date: date of operation
        :return: dictionary of all relevant trip data
        """
        # Get service IDs operating on the given date
        sids = self.get_service_ids(input_date)
        # Get only the relevant trips
        return self.get_trips_from_sids(sids, ref_date=input_date)

    def get_n_trips_per_day(self):
        # Identify busiest day
        dow_dict = {
            0: 'monday',
            1: 'tuesday',
            2: 'wednesday',
            3: 'thursday',
            4: 'friday',
            5: 'saturday',
            6: 'sunday'
        }

        sid_df = pd.DataFrame(
            self.trips_df.groupby('service_id')['trip_id'].nunique().rename('n_trips')
        )
        sid_df['n_blocks'] = self.trips_df.groupby('service_id')['block_id'].nunique()

        if self.calendar_df is not None:
            cal_df = self.calendar_df.copy()

            cal_df = cal_df.melt(
                cal_df[['service_id', 'start_date', 'end_date']]
            ).rename({'variable': 'weekday'}, axis=1)
            cal_df = cal_df[cal_df['value'] == 1].drop('value', axis=1)
            # Sometimes, all day values are 0 for some reason. In this case,
            # disregard calendar.txt and move on.
            if cal_df.empty:
                patterns = pd.DataFrame(
                    columns=['date', 'weekday', 'service_id']
                )
            else:
                # Now match up dates and SID values
                patterns = cal_df[['service_id', 'start_date', 'end_date', 'weekday']]
                patterns['date'] = [
                    pd.date_range(s, e, freq='d')
                    for s, e in zip(
                        patterns['start_date'],
                        patterns['end_date'])]
                patterns = patterns.explode('date').drop(['start_date', 'end_date'], axis=1)

                dates_ix = pd.date_range(
                    cal_df['start_date'].min(), cal_df['end_date'].max(), freq='D'
                )

                dates_df = pd.DataFrame(
                    data={
                        'date': dates_ix.to_numpy(),
                        'weekday': dates_ix.dayofweek.to_series().apply(lambda x: dow_dict[x])}
                )

                # Match dates with service IDs
                patterns = pd.merge(dates_df, patterns, on=['date', 'weekday'])

        else:
            patterns = pd.DataFrame(
                columns=['date', 'weekday', 'service_id']
            )

        # Incorporate exceptions from calendar_dates
        if self.calendar_dates_df is not None:
            cal_dates_df = self.calendar_dates_df.copy()
            service_add = cal_dates_df[cal_dates_df['exception_type'] == 1].drop(
                'exception_type', axis=1)
            service_rmv = cal_dates_df[cal_dates_df['exception_type'] == 2].drop(
                'exception_type', axis=1)

            # Add service IDs based on calendar_dates
            if len(service_add) > 0:
                service_add['weekday'] = service_add['date'].dt.dayofweek.apply(
                    lambda x: dow_dict[x])
                patterns = pd.concat([patterns, service_add], ignore_index=True)

            # Remove service IDs based on calendar_dates
            for _, row in service_rmv.sort_values(by='date').iterrows():
                match = (patterns['service_id'] == row['service_id']) & (
                        patterns['date'] == row['date'])
                patterns = patterns[~match]

        # Bring in number of trips and blocks based on service IDs
        patterns = pd.merge(patterns, sid_df, on='service_id')

        day_totals = patterns.groupby('date')[['n_trips']].sum()

        return day_totals

    @staticmethod
    def add_depot_deadhead(
            trips_df: pd.DataFrame, depot_lat: float, depot_lon: float
    ):
        """
        Add deadhead trips to and from depot. Note that these are not
        actually added as standalone trips; they just increase the
        distance of the first and last trips. For example, if the first
        trip on a block is 10 miles long, this method determines that
        5 miles of deadheading is required to reach the start of that
        trip from the depot and updates the trip distance to be 15
        miles.

        :param trips_df: DataFrame of trips
        :param depot_lat: depot latitude
        :param depot_lon: depot longitude
        :return:
        """
        trips_df = trips_df.set_index('trip_id')
        pullin_coords = list()
        pullout_coords = list()
        gb = trips_df.groupby('block_id')
        for block_id, subdf in gb:
            # Sort trips
            subdf = subdf.sort_values(by='trip_idx', ascending=True)

            # Get pull-out trip info
            pullout = subdf.iloc[0]
            pullout_coords.append(
                (pullout.loc['start_lat'], pullout.loc['start_lon'])
            )

            # Get pull-in trip info
            pullin = subdf.iloc[-1]
            pullin_coords.append(
                (pullin.loc['end_lat'], pullin.loc['end_lon'])
            )

        # Remove duplicates
        pullout_coords = list(set(pullout_coords))
        pullin_coords = list(set(pullin_coords))

        # Get OSM data
        dh_data = get_updated_osm_data(
            origins=[(depot_lat, depot_lon)] + pullin_coords,
            dests=[(depot_lat, depot_lon)] + pullout_coords
        )

        # Add to trip distances
        for block_id, subdf in gb:
            # Sort trips
            subdf = subdf.sort_values(by='trip_idx', ascending=True)
            # Add pull-out distance to first trip
            t0 = subdf.iloc[0]
            trips_df.loc[
                t0.name, 'total_dist'
            ] += dh_data[
                (depot_lat, depot_lon),
                (t0['start_lat'], t0['start_lon'])
            ]['distance']

            # Add pull-in distance to last trip
            tlast = subdf.iloc[-1]
            trips_df.loc[
                tlast.name, 'total_dist'
            ] += dh_data[
                (tlast['end_lat'], tlast['end_lon']),
                (depot_lat, depot_lon)
            ]['distance']

        return trips_df.reset_index()

    # TODO: might fit better as a function outside this class
    @staticmethod
    def build_opt_inputs(
        charging_blocks, trip_df, charge_nodes, charge_coords,
        depot_coords
    ):
        """
        Build inputs for optimization model.

        :return: dictionary of keyword arguments for ChargerLocationModel
        """
        trip_start_times = dict()
        trip_start_locs = dict()
        trip_end_times = dict()
        trip_end_locs = dict()
        trip_dists = dict()
        inter_trip_times = dict()
        inter_trip_dists = dict()
        # Earliest time any trip starts (so we can map time in minutes
        # from start back to time of day)
        zero_time = None

        block_gb = trip_df.groupby('block_id')
        for v in charging_blocks:
            block_df = block_gb.get_group(v)
            # Make extra sure trips are sorted in order
            block_df = block_df.sort_values(by='trip_idx')
            for i, t in enumerate(block_df.index):
                # Add 1 to the trip index to account for deadheading
                # from depot, which we model as trip 0
                t_idx = i + 1

                fmt_str = '%H:%M:%S'
                t_ref = datetime.strptime('00:00:00', fmt_str)
                if not zero_time or t_ref < zero_time:
                    zero_time = t_ref

                start_time = block_df.at[t, 'start_time']
                trip_start_times[v, t_idx] = \
                    (start_time - t_ref).total_seconds() / 60
                trip_start_locs[v, t_idx] = (
                    block_df.at[t, 'start_lat'], block_df.at[t, 'start_lon'])

                end_time = block_df.at[t, 'end_time']
                trip_end_times[v, t_idx] = \
                    (end_time - t_ref).total_seconds() / 60
                trip_end_locs[v, t_idx] = (
                    block_df.at[t, 'end_lat'], block_df.at[t, 'end_lon'])

                trip_dists[v, t_idx] = block_df.at[t, 'total_dist']

        # Calculate all necessary distances. Uses two calls, but file is saved
        # in between and reloaded so everything ends up in osm_charger_data.
        unique_start_locs = list(set(trip_start_locs.values()))
        unique_end_locs = list(set(trip_end_locs.values()))
        charger_locs = list(set(charge_coords.values()))

        osm_charger_data = get_dh_dict(
            unique_start_locs, unique_end_locs, charger_locs, depot_coords
        )

        trip_start_chg_dists = dict()
        trip_start_chg_times = dict()
        trip_end_chg_dists = dict()
        trip_end_chg_times = dict()
        for v in charging_blocks:
            block_df = block_gb.get_group(v)
            for i, t in enumerate(block_df.index):
                # Add 1 to the trip index to account for deadheading
                # from depot, which we model as trip 0
                t_idx = i + 1
                for s in charge_nodes:
                    osm_start = osm_charger_data[
                        charge_coords[s], trip_start_locs[v, t_idx]]
                    osm_end = osm_charger_data[
                        trip_end_locs[v, t_idx], charge_coords[s]]
                    trip_start_chg_dists[v, t_idx, s] = osm_start['distance']
                    trip_start_chg_times[v, t_idx, s] = osm_start['duration']
                    trip_end_chg_dists[v, t_idx, s] = osm_end['distance']
                    trip_end_chg_times[v, t_idx, s] = osm_end['duration']

                # TODO: update this condition to reflect depot trips?
                if i < len(block_df) - 1:
                    # Add inter-trip details if there are any trips
                    # remaining
                    osm_inter = osm_charger_data[
                        trip_end_locs[v, t_idx], trip_start_locs[v, t_idx + 1]]
                    inter_trip_dists[v, t_idx] = osm_inter['distance']
                    inter_trip_times[v, t_idx] = osm_inter['duration']

            # Add in trips to/from depot at end/start of day.
            osm_trip_0 = osm_charger_data[
                depot_coords, trip_start_locs[v, 1]]
            trip_dists[v, 0] = osm_trip_0['distance']
            trip_start_times[v, 0] = trip_start_times[v, 1]
            trip_end_times[v, 0] = trip_start_times[v, 1]
            inter_trip_dists[v, 0] = 0
            inter_trip_times[v, 0] = 0

            end_idx = t_idx + 1
            osm_last_trip = osm_charger_data[
                trip_end_locs[v, end_idx - 1], depot_coords]
            trip_dists[v, end_idx] = osm_last_trip['distance']
            # Maintain penalty for late arrival to depot
            trip_start_times[v, end_idx] = trip_end_times[v, end_idx - 1]
            trip_end_times[v, end_idx] = trip_end_times[v, end_idx - 1]
            inter_trip_dists[v, end_idx - 1] = 0
            inter_trip_times[v, end_idx - 1] = 0
            inter_trip_dists[v, end_idx] = 0
            inter_trip_times[v, end_idx] = 0

            for s in charge_nodes:
                # Add data for initial trip from depot
                # Don't charge until first real trip is complete (bus
                # fully charged when leaving depot anyway)
                trip_start_chg_dists[v, 0, s] = 1000
                trip_start_chg_times[v, 0, s] = 1000
                trip_end_chg_dists[v, 0, s] = 1000
                trip_end_chg_times[v, 0, s] = 1000

                # Add data for final trip to depot
                # Charging distance/time at the start of depot return
                # trip are the same as at end of final "real" trip
                trip_start_chg_dists[v, end_idx, s] = trip_end_chg_dists[
                    v, end_idx - 1, s]
                trip_start_chg_times[v, end_idx, s] = trip_end_chg_times[
                    v, end_idx - 1, s]
                # No sense in charging after final trip (to depot)
                trip_end_chg_dists[v, end_idx, s] = 1000
                trip_end_chg_times[v, end_idx, s] = 1000

        veh_trip_pairs = list(trip_start_times.keys())

        opt_kwargs = {
            'veh_trip_pairs': veh_trip_pairs,
            'trip_start_times': trip_start_times,
            'trip_end_times': trip_end_times,
            'trip_dists': trip_dists,
            'inter_trip_dists': inter_trip_dists,
            'trip_start_chg_dists': trip_start_chg_dists,
            'trip_end_chg_dists': trip_end_chg_dists,
            'inter_trip_times': inter_trip_times,
            'trip_start_chg_times': trip_start_chg_times,
            'trip_end_chg_times': trip_end_chg_times,
            'zero_time': zero_time}

        return opt_kwargs


def get_updated_osm_data(origins, dests, filename=None):
    """
    Update time/distance data for charging using Openrouteservice.
    Checks for existence of data to minimize unnecessary API calls.

    :param origins: Origin coordinates
    :param dests: Destination coordinates
    :param filename: String giving file name to check for existing data
        and write updated data
    :return: Dictionary of all charging data. Also pickles the updated
        dict for future use.
    """
    if filename is None:
        filename = str(Path(__file__).resolve().parent.parent /
                       'data' / 'osm' / 'osm_charge_data.pickle')

    # Read in saved dict of calculated costs
    try:
        with open(filename, 'rb') as handle:
            charging_travel_data = pickle.load(handle)
    except FileNotFoundError:
        # If file doesn't exist, create new dict
        charging_travel_data = dict()

    orig_osm = list()
    dest_osm = list()
    for org in origins:
        for dst in dests:
            if (org, dst) not in charging_travel_data:
                if org not in orig_osm:
                    orig_osm.append(org)
                if dst not in dest_osm:
                    dest_osm.append(dst)

    if orig_osm and dest_osm:
        n_dests = len(dest_osm)
        n_orig = len(orig_osm)
        osm_start = time.time()
        # Don't exceed OSM call size limit
        if n_dests * n_orig <= 3500:
            osm_dh_data = get_osm_distance(orig_osm, dest_osm)
            charging_travel_data = {**charging_travel_data, **osm_dh_data}
        else:
            if n_dests >= 3500:
                raise ValueError(
                    'Too many destinations to make any API calls.')
            n_orig_per_call = int(np.floor(3500 / n_dests))
            n_calls = int(np.ceil(n_orig / n_orig_per_call))
            logger.info(
                '{} unique route requests. {} ORS calls required.'.format(
                    n_dests*n_orig, n_calls))
            for c in range(n_calls):
                logger.info('API request {} underway.'.format(c+1))
                if c == n_calls-1:
                    part_origs = orig_osm[c*n_orig_per_call:]
                else:
                    part_origs = orig_osm[
                                 c*n_orig_per_call:(c+1)*n_orig_per_call]
                osm_dh_data = get_osm_distance(part_origs, dest_osm)
                charging_travel_data = {
                    **charging_travel_data, **osm_dh_data}
        logger.info(
            'OpenRouteService matrix returned in {:.2f} seconds.'.format(
                time.time() - osm_start))

        with open(filename, 'wb') as handle:
            pickle.dump(charging_travel_data, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    return charging_travel_data


def get_osm_distance(orig_list, dest_list):
    """
    Get all driving distances between provided coordinates using Openrouteservice.

    :param orig_list: List of origin coordinates, each of which is a
        tuple or list of (lat, lon) values
    :param dest_list: List of destination coordinates, each of which is
        a tuple or list of (lat, lon) values
    :return: Dict of distance and duration to drive from orig to dest,
        as calculated by OpenRouteService
    """
    # Openrouteservice setup
    try:
        config = dotenv_values(find_dotenv())
        ors_key = config['ORS_KEY']

    except KeyError:
        raise KeyError(
            'No Openrouteservice key found in .env file. For more information'
            ', please see the project README file.'
        )
    ors = client.Client(key=ors_key)

    # ORS uses reverse coordinate order (lon, then lat)
    orig_list_osm = [list(reversed(o)) for o in orig_list]
    dest_list_osm = [list(reversed(d)) for d in dest_list]
    all_coords = orig_list_osm + dest_list_osm
    orig_ix = list(range(len(orig_list_osm)))
    dest_ix = [len(orig_list_osm) + i for i in range(len(dest_list_osm))]
    logger.info('OSM request: {} origins, {} destinations ({} total '
                 'routes)'.format(len(orig_ix), len(dest_ix),
                                  len(orig_ix)*len(dest_ix)))
    params_matrix = {'profile': 'driving-hgv',
                     'metrics': ['duration', 'distance'],
                     'locations': all_coords,
                     'sources': orig_ix,
                     'destinations': dest_ix}
    res = ors.distance_matrix(**params_matrix)

    # Get the distance and time, converting from meters to miles and
    # from seconds to minutes
    out_dict = dict()
    for o_ix in orig_ix:
        for d_ix in range(len(dest_list)):
            try:
                dh_dict = {
                    'distance': res['distances'][o_ix][d_ix] / 1609,
                    'duration': res['durations'][o_ix][d_ix] / 60}
            except TypeError:
                # Sometimes ORS returns None for distance/duration. If
                # this happens, use Manhattan distance and 20 mph speed
                orig_lon, orig_lat = orig_list_osm[o_ix]
                dest_lon, dest_lat = dest_list_osm[d_ix]
                dh_dist = manhattan_np(
                    lon1=orig_lon, lat1=orig_lat, lon2=dest_lon, lat2=dest_lat
                )
                dh_time = dh_dist * 60 / 20
                dh_dict = {
                    'distance': dh_time,
                    'duration': dh_time
                }
            out_dict[tuple(orig_list[o_ix]), tuple(dest_list[d_ix])] = dh_dict

    return out_dict


def get_gmap_directions(orig, dest):
    """
    Get Google directions between a pair of points.

    :param orig: Origin coordinates, tuple of (lat, lon) values
    :param dest: Destination coordinates, tuple of (lat, lon) values
    :return: Directions to drive from orig to dest,
        as calculated by Google Maps, as encoded polyline
    """
    # Google API key
    try:
        config = dotenv_values(find_dotenv())
        gmaps_key = config['GMAPS_KEY']
    except KeyError:
        raise KeyError(
            'No Openrouteservice key found in .env file. For more information'
            ', please see the project README file.'
        )

    gmaps = googlemaps.Client(key=gmaps_key)
    directions = gmaps.directions(orig, dest, 'driving')
    return directions[0]['overview_polyline']['points']


def turn_over_datetime(dt_str):
    """
    Read a pseudo-datetime string in which the hours are greater than
    23, as sometimes happens with GTFS.

    :param dt_str: string of datetime formatted as %H:%M:%S
    :return: datetime object corresponding to given string
    """
    fmt_str = '%H:%M:%S'
    extra_hrs = int(dt_str[:2]) - 23
    dt_str = '23' + dt_str[2:]
    dt_obj = datetime.strptime(dt_str, fmt_str) \
             + timedelta(hours=extra_hrs)
    return dt_obj


def to_datetime_safe(srs, date):
    """
    Safely convert a Pandas Series of GTFS time strings to datetime
    objects. This custom function does is necessary because feeds often
    use a time string with > 24 hour formatting, e.g. 25:00:00 meaning
    1:00 a.m. the following day.

    :param srs: series to be formatted as datetime
    :param date: reference date prepended to all times
    """
    df = pd.DataFrame(srs)
    try:
        df[['hrs', 'min', 'sec']] = df[srs.name].str.split(
            ':', expand=True).astype(int)
    except ValueError:
        raise GTFSError(
            'time string not formatted properly, should be "%H:%M:%S"'
        )

    # Calculate seconds since midnight
    df['total_sec'] = 3600*df['hrs'] + 60*df['min'] + df['sec']
    df[srs.name] = pd.to_datetime(df['total_sec'], origin=date, unit='s')
    return df[srs.name]


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    credit: https://stackoverflow.com/a/29546836/8576714

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    miles = km / 1.609
    return miles


def manhattan_np(lon1, lat1, lon2, lat2):
    """
    Calculate the Manhattan (l1-norm) distance between two points
    on the earth (specified in decimal degrees)

    Works by calculating 2 haversine distances.

    """
    return haversine_np(lon1, lat1, lon2, lat1) + haversine_np(
        lon2, lat1, lon2, lat2
    )


def get_shape(shapes_df: pd.DataFrame, shape_id: int):
    """
    Get the shape of a trip (as a sequence of coordinates)
    :param shapes_df: DF of all trip shapes, from GTFS shapes.txt
    :param shape_id: shape ID requested
    :return: 2-tuple of pd.Series giving all longitudes (0) and
        latitudes (1) of points in shape
    """
    this_shape = shapes_df[shapes_df['shape_id'] == shape_id]
    this_shape.sort_values(by='shape_pt_sequence')
    return this_shape['shape_pt_lon'], this_shape['shape_pt_lat']


def get_dh_dict(
        trip_start_locs: list[tuple[float, float]],
        trip_end_locs: list[tuple[float, float]],
        charger_locs: list[tuple[float, float]],
        depot_coords: list[tuple[float, float]] | tuple[float, float] = None
) -> dict[
        tuple[tuple[float, float], tuple[float, float]],
        dict[str, float]
]:
    """
    Get a dictionary of deadhead distances and durations between the
    relevant input coordinates. This includes (1) deadhead from each
    trip to the next, (2) deadhead from each trip end to each charger,
    (3) deadhead form each charger to the start of each trip, and (4)
    deadhead to and from the depot.

    :param trip_start_locs: list of trip start coordinates
    :param trip_end_locs: list of trip end coordinates
    :param charger_locs: list of charger coordinates
    :param depot_coords: list or tuple of depot coordinates
    """
    # Calculate all necessary distances. Uses two calls, but file is saved
    # in between and reloaded so everything ends up in osm_charger_data.
    unique_start_locs = list(set(trip_start_locs))
    unique_end_locs = list(set(trip_end_locs))
    if depot_coords is not None:
        if type(depot_coords) == tuple:
            charger_locs = charger_locs + [depot_coords]
        elif type(depot_coords) == list:
            charger_locs = charger_locs + depot_coords
        else:
            raise TypeError('depot_coords must be list or tuple')
    # Get distances from trip ends to chargers
    # (note this function writes and reloads a file each time)
    _ = get_updated_osm_data(
        unique_end_locs, charger_locs)
    # Get distances from chargers to trip starts
    _ = get_updated_osm_data(
        charger_locs, unique_start_locs
    )
    # Get distances between trips
    osm_charger_data = get_updated_osm_data(
        unique_end_locs, unique_start_locs)
    return osm_charger_data
