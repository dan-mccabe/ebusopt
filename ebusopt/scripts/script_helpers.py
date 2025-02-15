from ebusopt.gtfs_beb import GTFSData
from ebusopt.gtfs_beb.data import get_updated_osm_data
import logging
import pandas as pd
import numpy as np
import scipy.stats as sps
from pathlib import Path

logger = logging.getLogger('script_helpers')

def build_trips_df(
    gtfs, date, routes, depot_coords, routes_60, route_method='exclusive',
    add_depot_dh=True, add_trip_dh=False, add_kwh_per_mi=False,
    add_durations=False, rng=None
):
    # Initialize GTFS
    day_trips = gtfs.get_trips_from_date(date)

    # Determine which blocks have trips exclusively on these routes
    gb = day_trips.groupby('block_id')
    route_blocks = list()
    for block_id, subdf in gb:
        if route_method == 'exclusive':
            include_block = all(subdf['route_short_name'].isin(routes))

        elif route_method == 'inclusive':
            include_block = any(subdf['route_short_name'].isin(routes))

        else:
            raise ValueError(
                'route_method must be either "exclusive" or "inclusive"'
            )

        if include_block:
            route_blocks.append(block_id)

    beb_trips = day_trips[
        day_trips['block_id'].isin(route_blocks)
    ]
    # Add all trip data columns (e.g. locations and distances)
    beb_trips = gtfs.add_trip_data(beb_trips, date)
    beb_trips['duration_sched'] = (
            beb_trips['end_time'] - beb_trips['start_time']
        ).dt.total_seconds() / 60

    beb_trips = beb_trips.rename(columns={'route_short_name': 'route'})
    block_types = beb_trips.groupby('block_id')['route'].unique().apply(
        lambda x: any(rt in routes_60 for rt in x)
    ).astype(int).rename('60_dummy')
    beb_trips = beb_trips.merge(
        block_types, left_on='block_id', right_index=True
    )

    if add_trip_dh:
        # Add deadhead to next trip
        beb_trips = GTFSData.add_deadhead(beb_trips)
        block_gb = beb_trips.groupby('block_id')
        dh_dfs = list()
        for block_id, block_df in block_gb:
            block_df = block_df.sort_values(by='trip_idx', ascending=True)

            # Associate DH with the next trip, since we assume charging
            # would happen before DH in charge scheduling approach.
            block_df['dh_dist'] = block_df['dh_dist'].shift(1).fillna(0)
            dh_dfs.append(block_df)
        beb_trips = pd.concat(dh_dfs)

    # Add pull-in and pull-out trip distances (note that scheduling
    # and charger location code handle these differently, and depot DH
    # should only be added with this approach for charge scheduling).
    if add_depot_dh:
        beb_trips = GTFSData.add_depot_deadhead(beb_trips, *depot_coords)

    if add_durations:
        # Add duration and kwh per mi
        beb_trips = add_realtime_durations(
            trips_to_lookup=beb_trips,
            realtime_summary=load_realtime_summary(),
            sim_all=False,
            rng=rng
        )
    if add_kwh_per_mi:
        beb_trips = predict_kwh_per_mi(beb_trips, rng=rng)

    # Optimization expects trips to be indexed from zero
    beb_trips['trip_idx'] -= 1

    return beb_trips


def build_charger_location_inputs(
    gtfs, trips_df, chargers_df, depot_coords, battery_cap
):
    charge_nodes = list(chargers_df['name'])
    charge_coords = dict()
    for n in charge_nodes:
        subdf = chargers_df[chargers_df['name'] == n]
        charge_coords[n] = (float(subdf['lat'].values[0]),
                            float(subdf['lon'].values[0]))

    all_blocks = list(trips_df['block_id'].unique())

    # Build inputs for optimization
    # This will give us all trip-specific data, but nothing
    # about vehicles or chargers (besides deadhead params)
    opt_kwargs = gtfs.build_opt_inputs(
        all_blocks, trips_df, charge_nodes,
        charge_coords, depot_coords)

    # Add charger details
    chargers_df = chargers_df.copy().set_index('name')
    fixed_costs = chargers_df['fixed_cost'].to_dict()
    charger_costs = chargers_df['charger_cost'].to_dict()
    max_ch = chargers_df['max_chargers'].to_dict()
    charger_kws = chargers_df['kw'].to_dict()
    # convert to kWh/min
    charger_rates = {s: charger_kws[s] / 60 for s in charger_kws}

    # Add vehicle details
    bus_caps = {b: battery_cap for b in all_blocks}
    # Dict of kWh per mile
    block_rate = trips_df.groupby('block_id')['kwh_per_mi'].mean().to_dict()
    energy_rates = {
        (v, t): block_rate[v] for (v, t) in opt_kwargs['veh_trip_pairs']
    }

    # Create full dict of optimization inputs
    opt_kwargs['vehicles'] = all_blocks
    opt_kwargs['chg_sites'] = charge_nodes
    opt_kwargs['chg_lims'] = bus_caps
    opt_kwargs['chg_rates'] = charger_rates
    opt_kwargs['energy_rates'] = energy_rates
    opt_kwargs['site_costs'] = fixed_costs
    opt_kwargs['charger_costs'] = charger_costs
    opt_kwargs['max_chargers'] = max_ch
    opt_kwargs['depot_coords'] = depot_coords
    opt_kwargs['trips_df'] = trips_df

    return opt_kwargs


def build_scheduling_inputs(
    beb_trips, chargers_df, u_max, energy_method='exact',
    duration_method='exact', energy_quantile=0.5, duration_quantile=0.5,
    dh_cutoff_dist=1, constant_val=3
):
    # Create a copy so we don't modify the input
    beb_trips = beb_trips.copy()
    beb_trips['kwh'] = beb_trips['kwh_per_mi'] * (
            beb_trips['total_dist'] + beb_trips['dh_dist']
    )
    block_energy = beb_trips.groupby('block_id')['kwh'].sum()
    opt_blocks = block_energy[block_energy > u_max].index.tolist()

    # only include blocks that need opportunity charging
    beb_trips = beb_trips[beb_trips['block_id'].isin(opt_blocks)].copy()

    if duration_method == 'quantile':
        rt_dist = pd.read_csv(
            '../data/processed/schedule_deviation_distributions.csv')
        # Only use the training dates
        rt_dist = rt_dist[rt_dist['date'] <= '2024-03-27']

    # Initialize parameter dicts
    delta = dict()
    sigma = dict()
    tau = dict()
    max_chg_time = dict()

    # Track terminal coords to get dist to chargers
    term_coords = list()

    # Set reference time (midnight on first day observed)
    t_ref = pd.to_datetime(beb_trips['start_time'].dt.date).min()

    # Constants for handling time
    # '7160963'
    for ix, rw in beb_trips.iterrows():
        dict_ix = (rw['block_id'], rw['trip_idx'])
        start_time = rw['start_time']

        sigma[dict_ix] = (start_time - t_ref).total_seconds() / 60

        if energy_method == 'quantile':
            # Create a normal distribution that represents this trip's
            # kWh per mile
            kwh_per_mi_dist = sps.norm(
                loc=rw['kwh_per_mi_mean'], scale=rw['kwh_per_mi_err']
            )
            # Sample the input quantile from that distribution
            kwh_per_mi = kwh_per_mi_dist.ppf(energy_quantile)

        elif energy_method == 'mean':
            kwh_per_mi = rw['kwh_per_mi_mean']

        elif energy_method == 'exact':
            kwh_per_mi = rw['kwh_per_mi']

        elif energy_method == 'constant':
            kwh_per_mi = constant_val

        else:
            raise ValueError(
                'method must be "constant", "quantile," "mean", or "exact".'
            )
        # Set energy consumption for this trip
        delta[dict_ix] = kwh_per_mi * (rw['total_dist'] + rw['dh_dist'])

        if duration_method == 'quantile':
            # Create a normal distribution that represents this trip's
            # duration
            rt_dev = rt_dist.groupby('route').get_group(
                rw['route'])['duration_difference_pct'].quantile(
                duration_quantile)
            duration = (1 + rt_dev / 100) * rw['duration_sched']

        elif duration_method == 'mean':
            duration = rw['duration_mean']

        elif duration_method == 'exact':
            duration = rw['duration']

        elif duration_method == 'scheduled':
            duration = rw['duration_sched']

        else:
            raise ValueError(
                'method must be "scheduled", "quantile," "mean", or "exact".'
            )

        # Set the trip duration parameter based on the supplied method
        tau[dict_ix] = duration

        if (rw['end_lon'], rw['end_lat']) not in term_coords:
            term_coords.append((rw['end_lat'], rw['end_lon']))

    # Set charging upper bound
    # First, get all DH distances
    charger_coords = list(
        zip(
            chargers_df['lat'].tolist(),
            chargers_df['lon'].tolist()
        )
    )
    charger_dh = get_updated_osm_data(term_coords, charger_coords)
    # chargers_df = chargers_df.copy().set_index('name')
    # Use it to set charging limit
    for ix, rw in beb_trips.iterrows():
        for c in chargers_df.index:
            if charger_dh[
                (rw['end_lat'], rw['end_lon']),
                (chargers_df.loc[c, 'lat'], chargers_df.loc[c, 'lon'])
            ]['distance'] < dh_cutoff_dist:
                max_chg_time[c, rw['block_id'], rw['trip_idx']] = \
                    60 * u_max / chargers_df.loc[c, 'kw']

            else:
                max_chg_time[c, rw['block_id'], rw['trip_idx']] = 0

    case_data = {
        'sigma': sigma,
        'tau': tau,
        'delta': delta,
        'max_chg_time': max_chg_time
    }

    return case_data


def build_sim_inputs(
    opt_df, beb_trips, depot_coords, min_soc, max_soc, battery_kwh
):
    chg_plan_df = opt_df[
        ['block_id', 'trip_idx', 'charger', 'chg_time']
    ].copy()
    chg_plan_df = chg_plan_df[
        chg_plan_df['chg_time'] > 0
        ]
    logger.info(
        '{} charges scheduled for {} charging blocks, totaling '
        '{:.2f} minutes'.format(
            len(chg_plan_df), chg_plan_df['block_id'].nunique(),
            chg_plan_df['chg_time'].sum()
        )
    )
    chg_plan_df.set_index(
        ['block_id', 'trip_idx', 'charger'],
        inplace=True
    )

    depot_df = pd.DataFrame.from_dict(
        {'South Base': {'lat': depot_coords[0], 'lon': depot_coords[1]}},
        orient='index'
    )
    vehicles_df = pd.DataFrame(
        index=list(beb_trips['block_id'].unique())
    )
    vehicles_df['min_kwh'] = min_soc * battery_kwh
    vehicles_df['max_kwh'] = max_soc * battery_kwh

    # Build simulation inputs
    required_cols = [
        'block_id', 'trip_idx', 'trip_id', 'route', 'start_time', 'end_time',
        'start_lat', 'start_lon', 'end_lat', 'end_lon', 'total_dist',
        'dh_dist', 'duration_sched', 'duration', 'kwh_per_mi'
    ]
    optional_cols = [
        'kwh_per_mi_mean', 'kwh_per_mi_err', 'duration_mean', 'duration_err'
    ]
    optional_cols_incl = [c for c in optional_cols if c in beb_trips.columns]
    incl_cols = required_cols + optional_cols_incl
    trips_sim = beb_trips[incl_cols].set_index(
        ['block_id', 'trip_idx']
    )
    trips_sim['duration'] = pd.to_timedelta(trips_sim['duration'], unit='min')

    return dict(
        trip_data_df=trips_sim,
        chg_plan_df=chg_plan_df,
        vehicles_df=vehicles_df,
        depot_df=depot_df
    )


def add_realtime_durations(
        trips_to_lookup, realtime_summary, sim_all=False, rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    trips_to_lookup.reset_index(inplace=True)
    date = pd.to_datetime(trips_to_lookup['start_time'].dt.date.min())
    trips_to_lookup['date'] = date
    trips_to_lookup['trip_id'] = trips_to_lookup['trip_id'].astype(str)
    # Merge in realtime duration
    trips_to_lookup = trips_to_lookup.merge(
        realtime_summary[['date', 'trip_id', 'duration_rt']],
        on=['date', 'trip_id'], how='left'
    ).rename(columns={'duration_rt': 'duration'})

    # Read in schedule deviation info
    filepath = Path(__file__).resolve().parent.parent / 'data' / 'processed' \
        / 'schedule_deviation_distributions.csv'
    dur_df = pd.read_csv(
        filepath,
        parse_dates=['date'],
        dtype={'trip_id': str, 'route': str}
    )
    dur_df = dur_df[
        dur_df['route'].isin(trips_to_lookup['route'].unique())
    ].copy()
    route_dur_gb = dur_df.groupby('route')['duration_difference_pct']
    dur_by_route = pd.DataFrame(
        data={
            'duration_diff_mean': route_dur_gb.mean(),
            'duration_diff_std': route_dur_gb.std()
        }
    )
    trips_to_lookup = trips_to_lookup.merge(
        dur_by_route, on='route', how='left'
    )
    # Add mean and standard deviation, which will be used by simulation
    trips_to_lookup['duration_mean'] = \
        trips_to_lookup['duration_sched'] \
        * (1 + trips_to_lookup['duration_diff_mean'] / 100)
    trips_to_lookup['duration_err'] = \
        trips_to_lookup['duration_sched'] \
        * (trips_to_lookup['duration_diff_std'] / 100)

    if sim_all:
        trips_to_lookup['duration'] = rng.normal(
            loc=trips_to_lookup['duration_diff_mean'],
            scale=trips_to_lookup['duration_diff_std']
        )
        trips_to_lookup['duration_src'] = 'simulated'
        return trips_to_lookup

    # Pick out the trips where we couldn't get the real duration
    trips_na = trips_to_lookup[trips_to_lookup['duration'].isna()].copy()
    # Simulate duration based on the realtime trip data
    trips_na['duration'] = trips_na['duration_sched'] * (
        1 + rng.normal(
            loc=trips_na['duration_diff_mean'],
            scale=trips_na['duration_diff_std'],
            size=len(trips_na)
        ) / 100
    )
    trips_na['duration_src'] = 'simulated'

    # Pick out the trips where we did find the real duration
    trips_dur = trips_to_lookup[~trips_to_lookup['duration'].isna()].copy()
    # Convert seconds to minutes
    trips_dur['duration'] /= 60
    trips_dur['duration_src'] = 'realtime'

    return pd.concat([trips_dur, trips_na])


def predict_kwh_per_mi(trips_df, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    elev_df = pd.read_csv(
        '../data/processed/elevation_by_shape_usgs.csv',
        dtype={'shape_id': str}
    )
    trips_df = trips_df.reset_index().merge(
        elev_df[['shape_id', 'elev_gain', 'elev_loss']], on='shape_id'
    )
    trips_df['gain_per_mi'] = trips_df['elev_gain'] / trips_df['total_dist']
    trips_df['loss_per_mi'] = trips_df['elev_loss'] / trips_df['total_dist']
    trips_df['mph'] = 60 * trips_df['total_dist'] / trips_df['duration_sched']
    weather_df = pd.read_csv(
        '../data/weather/full_2024_temperature.csv',
        parse_dates=['DATE'])
    weather_df['date_hour'] = pd.to_datetime(
        weather_df['DATE'].dt.date) + pd.to_timedelta(
        weather_df['DATE'].dt.hour, unit='h'
    )
    weather_df['HourlyDryBulbTemperature'] = weather_df[
        'HourlyDryBulbTemperature'].ffill()
    # Take the median recorded value for each hour
    merge_df = weather_df.rename(
        columns={'HourlyDryBulbTemperature': 'temperature'}).groupby(
        'date_hour')['temperature'].median()
    trips_df['start_hour'] = trips_df['start_time'].dt.round('h')
    # Merge in temperature
    trips_df = trips_df.merge(
        merge_df, left_on='start_hour', right_on='date_hour',
        how='left'
    )
    trips_df['heating_degrees'] = np.maximum(65 - trips_df['temperature'], 0)
    if '60_dummy' not in trips_df.columns:
        logger.info(
            'No vehicle size information provided for energy prediction. '
            'Defaulting to all 40-foot buses.'
        )
        trips_df['60_dummy'] = 0
    trips_df['heating_degrees_60'] = trips_df['60_dummy'] * trips_df[
        'heating_degrees']
    trips_df['heating_degrees_40'] = (1 - trips_df['60_dummy']) * trips_df[
        'heating_degrees']
    trips_df['kwh_per_mi_mean'] = 2.1384 + 0.6948 * trips_df['60_dummy'] \
        + 0.0466 * trips_df['gain_per_mi'] - 0.0351 * trips_df['loss_per_mi']\
        + 0.0248 * trips_df['heating_degrees_40'] \
        + 0.0375 * trips_df['heating_degrees_60'] - 0.0232 * trips_df['mph']
    trips_df['kwh_per_mi_err'] = 0.3424
    trips_df['kwh_per_mi'] = trips_df['kwh_per_mi_mean'] + rng.normal(
        loc=0, scale=0.3424, size=len(trips_df)
    )

    # Drop columns that were only needed for prediction
    trips_df.drop(
        columns=['gain_per_mi', 'loss_per_mi', 'mph', 'heating_degrees',
                 'heating_degrees_40', 'heating_degrees_60'],
        inplace=True
    )

    return trips_df


def load_realtime_summary():
    filepath = Path(__file__).resolve().parent.parent / 'data' / 'processed' \
        / 'cleaned_mar_24_realtime_all_trips.csv'
    df = pd.read_csv(
        filepath,
        dtype={'vehicle_id': str, 'trip_id': str}
    )
    df['date'] = pd.to_datetime(df['date'])
    return df



