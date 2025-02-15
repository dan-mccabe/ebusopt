from ebusopt.scripts.script_helpers import build_trips_df, \
    build_charger_location_inputs, build_scheduling_inputs, \
    build_sim_inputs
from ebusopt.opt.heuristic_charge_scheduling import repeat_heuristic
from ebusopt.opt.simulation import SimulationBatch
from ebusopt.opt.charger_location import ChargerLocationModel
from ebusopt.gtfs_beb import GTFSData
from ebusopt.vis import plot_trips_and_terminals, plot_deadhead
from pathlib import Path
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_charger_df_from_results(clm, charger_df):
    # Build dataframe of charger locations
    sched_chargers = charger_df.loc[
        charger_df['name'].isin(clm.opt_stations)
    ]

    # Sites we need to duplicate
    dup_chargers = {
        s: clm.num_chargers[s] for s in clm.opt_stations
        if clm.num_chargers[s] >= 2
    }

    next_ix = sched_chargers.index.max() + 1
    for s in dup_chargers:
        for ix in range(1, dup_chargers[s] + 1):
            # Get the original row
            s_rw = sched_chargers[sched_chargers['name'] == s].iloc[0]
            s_rw['name'] = s_rw['name'] + f' {ix}'
            sched_chargers.loc[next_ix] = s_rw.tolist()
            next_ix += 1
        # Drop the original row
        s_ix = sched_chargers[sched_chargers['name'] == s].index[0]
        sched_chargers.drop(s_ix, inplace=True)

    return sched_chargers


def test_charge_scheduling(random_seed):
    """
    Test charge scheduling performance using 3S heuristic across 4 test
    dates in March 2024. Some key parameters are hard-coded below, so
    be careful.

    In this function, we first run the charger location optimization
    model to select charging sites. These determine the charger location
    inputs to the scheduling optimization across the 4 test days, where
    we design a schedule using 3S based on the energy and duration
    quantile values specified below.
    """
    # Create numpy random number generator
    rng = np.random.default_rng(random_seed)

    # ---- Define routes and bus parameters ----
    gtfs_dir = '../data/gtfs/metro_mar24'
    ocl_date = datetime.datetime(2024, 3, 28)

    # All routes included in analysis
    beb_routes = [
        101, 102, 105, 106, 107, 131, 132,
        150, 153, 156, 160, 161, 162, 165,
        168, 177, 182, 183, 187, 193, 240
    ]
    beb_routes = [str(r) for r in beb_routes]

    # Routes operated by 60-foot buses
    routes_60 = [
        101, 102, 131, 132, 150, 162, 177, 193
    ]
    routes_60 = [str(r) for r in routes_60]

    # Depot at South Base
    depot_coords = (47.495809, -122.286190)

    # Load GTFS data
    gtfs = GTFSData.from_dir(gtfs_dir)

    # ---- Parameters for charger location ----
    # CSV file giving candidate charger sites
    site_fname = '../data/so_king_cty_sites.csv'
    # Power output of each charger
    chg_pwrs = 450 / 60
    # Maximum number of chargers per site
    n_max = 4
    # Cost parameters
    s_cost = 200000
    c_cost = 698447
    alpha = 190 * 365 * 12 / 60

    # Load candidate charging sites given by Metro and add params
    loc_df = pd.read_csv(site_fname)
    loc_df['max_chargers'] = n_max
    loc_df['kw'] = chg_pwrs * 60
    loc_df['fixed_cost'] = s_cost
    loc_df['charger_cost'] = c_cost

    # Bus parameters
    # Battery capacity in kWh
    battery_cap = 525 * 0.8

    beb_trips = build_trips_df(
        gtfs=gtfs,
        date=ocl_date,
        routes=beb_routes,
        routes_60=routes_60,
        depot_coords=depot_coords,
        add_depot_dh=True,
        add_kwh_per_mi=False,
        add_durations=False,
        rng=rng
    )
    logging.info(
        '{}: There are {} total trips to be served by {} BEB blocks.'.format(
            ocl_date.strftime('%m/%d'), len(beb_trips), beb_trips['block_id'].nunique()
        )
    )

    # Map of the problem instance
    inst_map = plot_trips_and_terminals(
        beb_trips, loc_df, gtfs.shapes_df, 'light'
    )
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'scale': 3
        }
    }
    show_map = False
    if show_map:
        inst_map.show(config=config)

    beb_trips['kwh_per_mi'] = 2.31 + (3.19 - 2.31) * beb_trips['60_dummy']

    opt_kwargs = build_charger_location_inputs(
        gtfs=gtfs,
        trips_df=beb_trips,
        chargers_df=loc_df,
        depot_coords=depot_coords,
        battery_cap=battery_cap
    )

    clm = ChargerLocationModel(**opt_kwargs)
    clm.solve(
        alpha=alpha, opt_gap=0, n_runs_heur=30,
        bu_kwh=battery_cap
    )
    clm.log_results()

    if show_map:
        fig = plot_deadhead(
            result_df=clm.to_df(), loc_df=loc_df, coords_df=beb_trips
        )
        fig.show()

    # ---- Process charger location results ----
    sched_chargers = get_charger_df_from_results(clm, loc_df)
    charger_list = list(sched_chargers['name'].unique())
    sched_chargers['n_chargers'] = 1
    # sched_chargers.to_csv('../data/optimized_charger_locs.csv')
    sched_chargers = sched_chargers.set_index('name')

    # ---- Run charge scheduling ----
    rho = {s: chg_pwrs for s in charger_list}

    for day in [28, 29, 30, 31]:
        current_dt = datetime.datetime(2024, 3, day)
        logging.info('\n---- Testing March {} ----'.format(day))
        # Create inputs for charge scheduling algorithm
        beb_trips = build_trips_df(
            gtfs=gtfs,
            date=current_dt,
            routes=beb_routes,
            routes_60=routes_60,
            depot_coords=depot_coords,
            add_depot_dh=True,
            add_trip_dh=True,
            add_durations=True,
            add_kwh_per_mi=True,
            rng=rng
        )

        # Filter out a few problematic blocks (most of these are cases
        # where a bus only completes our selected routes in the
        # direction where are charger is not located at the end)
        bad_blocks = [
            '7160811', '7160812', '7160812', '7160813', '7160807', '7160707']
        beb_trips = beb_trips[~beb_trips['block_id'].isin(bad_blocks)]

        logging.info(
            '{}: There are {} total trips to be served by {} BEB blocks.'.format(
                current_dt.strftime('%m/%d'), len(beb_trips),
                beb_trips['block_id'].nunique()
            )
        )

        case_data = build_scheduling_inputs(
            beb_trips=beb_trips, chargers_df=sched_chargers, u_max=battery_cap,
            energy_method='quantile', duration_method='quantile',
            energy_quantile=0.6, duration_quantile=0.9,
            dh_cutoff_dist=1.5
        )

        soln_df = repeat_heuristic(
            case_data, charger_list, rho, battery_cap, n_runs=30,
            random_mult=1, return_type='df'
        )
        # Indexing is inconsistent between simulation and optimization
        # soln_df['trip_idx'] += 1

        # ---- Run simulations ----
        sim_dict = build_sim_inputs(
            opt_df=soln_df,
            beb_trips=beb_trips,
            depot_coords=depot_coords,
            min_soc=0.15,
            max_soc=0.95,
            battery_kwh=battery_cap / 0.8
        )

        # Test performance using the actually observed realtime durations
        logging.info('-- Batch simulation with realtime durations --')
        batch = SimulationBatch(
            chargers_df=sched_chargers, ignore_deadhead=False, n_sims=100,
            vary_energy=True, vary_duration=False, seed=random_seed+1,
            **sim_dict
        )
        batch.run()
        batch.process_results()
        logging.info('exog delays: {}'.format(batch.exog_delay[0]))
        logging.info('mean % delayed trips: {:.2f}'.format(
            batch.pct_trips_delayed.mean()))
        logging.info('max unplanned charges: {}'.format(
            batch.n_unplanned_charges.max()))

        # print('-- Batch simulation with random durations --')
        # batch = SimulationBatch(
        #     chargers_df=sched_chargers, ignore_deadhead=False, n_sims=100,
        #     vary_energy=True, vary_duration=True, seed=100, **sim_dict
        # )
        # batch.run()
        # batch.process_results()
        # # print('exog delays:', batch.exog_delay)
        # print('max unplanned charges:', max(batch.n_unplanned_charges))


def run_quantile_sensitivity(random_seed):
    # Create numpy random number generator for repeatability
    rng = np.random.default_rng(random_seed)

    # ---- Define routes and bus parameters ----
    gtfs_dir = '../data/gtfs/metro_mar24'

    beb_routes = [
        101, 102, 105, 106, 107, 131, 132, 150, 153, 156, 160, 161, 162,
        165, 168, 177, 182, 183, 187, 193, 240
    ]
    beb_routes = [str(r) for r in beb_routes]

    # Routes operated by 60-foot buses
    routes_60 = [
        101, 102, 131, 132, 150, 162, 177, 193
    ]
    routes_60 = [str(r) for r in routes_60]

    # Depot at South Base
    depot_coords = (47.495809, -122.286190)

    # Load GTFS data
    gtfs = GTFSData.from_dir(gtfs_dir)

    # Power output of each charger
    charger_kw = 150

    # Bus parameters
    # Battery capacity in kWh
    battery_cap = 525 * 0.8

    # ---- Run charge scheduling ----
    sched_chargers = pd.read_csv(
        '../data/optimized_charger_locs.csv').set_index('name')
    sched_chargers['kw'] = charger_kw
    charger_list = sched_chargers.index.unique().tolist()
    rho = {s: charger_kw / 60 for s in charger_list}

    date_range = pd.date_range(start='4/1/24', end='4/30/24')
    sim_delays = dict()
    unplanned_chgs = dict()
    for date in date_range:
        # current_dt = datetime.datetime(2024, 3, day)
        logging.info('\n---- Testing {} ----'.format(date.strftime('%m/%d')))
        # Create inputs for charge scheduling algorithm
        beb_trips = build_trips_df(
            gtfs=gtfs,
            date=date,
            routes=beb_routes,
            routes_60=routes_60,
            depot_coords=depot_coords,
            add_depot_dh=True,
            add_trip_dh=True,
            add_durations=True,
            add_kwh_per_mi=True,
            rng=rng
        )

        # Filter out some problematic blocks that make some scenarios
        # infeasible. The results are more easily interpretable if
        # we just exclude these from the analysis.
        bad_blocks = [
            '7160811', '7160812', '7160813', '7160807', '7160707',
            '7160963', '7157686', '7157747', '7160851', '7117813',
            '7117813', '7117814', '7160814'
        ]
        beb_trips = beb_trips[~beb_trips['block_id'].isin(bad_blocks)]

        logging.info(
            '{}: There are {} total trips to be served by {} BEB blocks.'.format(
                date.strftime('%m/%d'), len(beb_trips),
                beb_trips['block_id'].nunique()
            )
        )

        for q in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            logging.info(
                '--- Testing Duration Quantile: {} ---'.format(q)
            )
            case_data = build_scheduling_inputs(
                beb_trips=beb_trips, chargers_df=sched_chargers, u_max=battery_cap,
                energy_method='quantile', duration_method='quantile',
                energy_quantile=0.5, duration_quantile=q,
                dh_cutoff_dist=1.5
            )

            soln_df = repeat_heuristic(
                case_data, charger_list, rho, battery_cap, n_runs=30,
                random_mult=1, return_type='df'
            )

            # ---- Run simulations ----
            sim_dict = build_sim_inputs(
                opt_df=soln_df,
                beb_trips=beb_trips,
                depot_coords=depot_coords,
                min_soc=0.15,
                max_soc=0.95,
                battery_kwh=battery_cap / 0.8,
            )

            # Test performance across different simulated durations
            logging.info('-- Batch simulation with simulated durations --')
            batch = SimulationBatch(
                chargers_df=sched_chargers, ignore_deadhead=False, n_sims=10,
                vary_energy=True, vary_duration=True, seed=random_seed+1,
                **sim_dict
            )
            batch.run()
            batch.process_results()
            if q in sim_delays:
                sim_delays[q] = np.concatenate(
                    (sim_delays[q], batch.charging_delay)
                )
                unplanned_chgs[q] = np.concatenate(
                    (unplanned_chgs[q], batch.n_unplanned_charges)
                )

            else:
                sim_delays[q] = batch.charging_delay
                unplanned_chgs[q] = batch.n_unplanned_charges

    names = [
        'Charging-Induced Delay (minutes)', 'Number of Unplanned Charges'
    ]
    for ix, d in enumerate(
            [sim_delays, unplanned_chgs]
    ):
        # Boxplot of total delay
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_ylabel(names[ix])
        ax.set_xlabel('Duration Quantile Used in Schedule Optimization')

        ax.boxplot(
            list(d.values()), tick_labels=d.keys()
        )
        plt.tight_layout()

        plt.savefig(
            '{}_duration_quantile_impacts_{}.pdf'.format(
                charger_kw, ix), dpi=300
        )
        plt.show()


def run_mar_28_scheduling(random_seed):
    # Create numpy random number generator for repeatability
    rng = np.random.default_rng(random_seed)

    # ---- Define routes and bus parameters ----
    gtfs_dir = Path(__file__).resolve().parent.parent / 'data' / 'gtfs' / 'metro_mar24'
    # gtfs_dir = '../data/gtfs/metro_mar24'

    beb_routes = [
        101, 102, 105, 106, 107, 131, 132, 150, 153, 156, 160, 161, 162,
        165, 168, 177, 182, 183, 187, 193, 240
    ]
    beb_routes = [str(r) for r in beb_routes]

    # Routes operated by 60-foot buses
    routes_60 = [
        101, 102, 131, 132, 150, 162, 177, 193
    ]
    routes_60 = [str(r) for r in routes_60]

    # Depot at South Base
    depot_coords = (47.495809, -122.286190)

    # Load GTFS data
    gtfs = GTFSData.from_dir(gtfs_dir)

    # Power output of each charger
    charger_kw = 450

    # Bus parameters
    # Battery capacity in kWh
    battery_cap = 525 * 0.8

    # ---- Run charge scheduling ----
    sched_chargers = pd.read_csv(
        '../data/optimized_charger_locs.csv').set_index('name')
    sched_chargers['kw'] = charger_kw
    charger_list = sched_chargers.index.unique().tolist()
    rho = {s: charger_kw / 60 for s in charger_list}
    date = datetime.datetime(2024, 3, 28)

    # Create inputs for charge scheduling algorithm
    beb_trips = build_trips_df(
        gtfs=gtfs,
        date=date,
        routes=beb_routes,
        routes_60=routes_60,
        depot_coords=depot_coords,
        add_depot_dh=True,
        add_trip_dh=True,
        add_durations=True,
        add_kwh_per_mi=False,
        rng=rng
    )
    beb_trips['kwh_per_mi'] = 2.31 + (3.19 - 2.31) * beb_trips['60_dummy']

    # Filter out some problematic blocks that make some scenarios
    # infeasible. The results are more easily interpretable if
    # we just exclude these from the analysis.
    bad_blocks = [
        '7160811', '7160812', '7160812', '7160813', '7160807', '7160707']
    beb_trips = beb_trips[~beb_trips['block_id'].isin(bad_blocks)]

    logging.info(
        '{}: There are {} total trips to be served by {} BEB blocks.'.format(
            date.strftime('%m/%d'), len(beb_trips),
            beb_trips['block_id'].nunique()
        )
    )

    case_data = build_scheduling_inputs(
        beb_trips=beb_trips, chargers_df=sched_chargers, u_max=battery_cap,
        energy_method='exact', duration_method='scheduled',
        dh_cutoff_dist=1.5
    )

    soln_df = repeat_heuristic(
        case_data, charger_list, rho, battery_cap, n_runs=1,
        random_mult=1, return_type='df'
    )

    soln_df.to_csv('mar_28_scheduling_result.csv')


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    # test_charge_scheduling(random_seed=115)
    run_quantile_sensitivity(random_seed=115)
    # run_mar_28_scheduling(random_seed=17)
