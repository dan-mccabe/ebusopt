"""
This script was originally created to run the analysis for our TR Part
C paper. It currently results in errors because of some more recent
changes to model building functions. It will need to be cleaned up if
the analysis is to be run again.
"""

import pandas as pd
from ebusopt.opt.charger_location import ChargerLocationModel
from ebusopt.opt.simulation import SimulationRun
from ebusopt.gtfs_beb import GTFSData
from ebusopt.scripts.script_helpers import build_trips_df, \
    build_charger_location_inputs
from ebusopt.vis import plot_trips_and_terminals
import logging
import datetime


def run_facility_location(
        route_list, site_file, battery_cap, kwh_per_mile, charge_power,
        site_cost, charger_cost, alpha, max_chargers, depot_coords,
        save_fname=None, summary_fname=None, opt_gap=None,
        gtfs_dir='../data/gtfs/metro_2020',
        test_date=datetime.datetime(2020, 3, 9)):
    # Read in all the data
    gtfs = GTFSData.from_dir(gtfs_dir)

    # Select routes that will be served by 60' BEBs
    if route_list == 'all':
        # Exclude any non-bus routes
        matching_routes = gtfs.filter_df(gtfs.routes_df, 'route_type', 3)
        all_rts = matching_routes['route_short_name'].unique().tolist()

        # Don't include trolleybuses either
        trolley_rts = [
            1, 2, 3, 4, 7, 10, 12, 13, 14, 36, 43, 44, 48, 49, 70]
        trolley_rts = [str(r) for r in trolley_rts]
        route_list = [r for r in all_rts if r not in trolley_rts]

    else:
        route_list = [str(r) for r in route_list]

    all_trips_df = build_trips_df(
        gtfs=gtfs,
        date=test_date,
        routes=route_list,
        depot_coords=depot_coords,
        route_method='exclusive',
        add_depot_dh=False,
        routes_60=route_list
    )
    # Set kWh per mile
    all_trips_df['kwh_per_mi'] = kwh_per_mile

    print('Number of trips on selected routes: {}'.format(len(all_trips_df)))

    all_blocks = all_trips_df['block_id'].unique()
    print('Number of blocks: {}'.format(len(all_blocks)))

    # Add all necessary fields
    all_trips_df.to_csv('../data/all_trip_data.csv')

    # Load candidate charging sites given by Metro
    loc_df = pd.read_csv(site_file)
    loc_df['max_chargers'] = max_chargers
    loc_df['kw'] = charge_power
    loc_df['fixed_cost'] = site_cost
    loc_df['charger_cost'] = charger_cost

    opt_kwargs = build_charger_location_inputs(
        gtfs=gtfs,
        trips_df=all_trips_df,
        chargers_df=loc_df,
        depot_coords=depot_coords,
        battery_cap=battery_cap
    )

    flm = ChargerLocationModel(**opt_kwargs)
    # Identify layover blocks
    flm.check_charging_needs()
    flm.check_feasibility()
    # Map trips and terminals (layover blocks only)
    lo_blocks = [v for v in flm.charging_vehs if v not in flm.infeas_vehs]
    layover_trips_df = all_trips_df[all_trips_df['block_id'].isin(lo_blocks)]
    inst_map = plot_trips_and_terminals(
        trips_df=layover_trips_df, locs_df=loc_df, shapes_df=gtfs.shapes_df)
    inst_map.show()

    flm.solve(
        alpha=alpha, opt_gap=opt_gap, n_runs_heur=100, bu_kwh=battery_cap
    )
    flm.log_results()
    flm.plot_chargers('../../results/metro_util.pdf')
    flm.plot_conflict_sets('../../results/metro_conflict_hist.pdf')

    if save_fname is not None:
        flm.to_csv(save_fname)

    if summary_fname is not None:
        flm.summary_to_csv(summary_fname)
    return flm


def run_trc_case():
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("beb_model").setLevel(logging.DEBUG)

    # Define inputs
    depot_coords = (47.495809, -122.286190)
    # CSV file giving candidate charger sites
    # site_fname = '../data/so_king_cty_sites.csv'
    site_fname = '../data/king_cty_sites_trc.csv'

    busiest_routes = [
        'A Line', 'C Line', 'D Line', 'E Line', 5, 8, 40, 41, 45, 62,
        65, 67, 106, 120, 372]
    # busiest_routes = [
    #     'A Line', 'C Line', 'D Line', 'E Line']
    # 40-ft routes from interim base
    # Source: Metro BEB Implementation report, 2020
    interim_40 = [22, 116, 153, 154, 156, 157, 158, 159, 168, 169, 177, 179,
                  180, 181, 182, 183, 186, 187, 190, 192, 193, 197]
    # 60-ft routes from interim base
    interim_60 = [101, 102, 111, 116, 143, 150, 157, 158, 159, 177, 178, 179,
                  180, 190, 192, 193, 197]
    interim_rts = list(set(interim_40 + interim_60))

    # Battery capacity in kWh
    battery_cap = 466 * 0.75
    # Energy consumption rate in kWh per mile
    kwh_per_mile = 3
    # Power output of each charger
    chg_pwrs = 450 / 60
    # Charger construction cost
    s_cost = 200000
    c_cost = 698447

    n_max = 4
    alpha = 190 * 365 * 12 / 60

    flm = run_facility_location(
        route_list=interim_60, site_file=site_fname, battery_cap=battery_cap,
        kwh_per_mile=kwh_per_mile, charge_power=chg_pwrs, site_cost=s_cost,
        charger_cost=c_cost, alpha=alpha, max_chargers=n_max, opt_gap=0.01,
        depot_coords=depot_coords, pickle_fname='../data/metro_inputs.pickle',
        save_fname='../../results/metro_case_results.csv',
        summary_fname='../../results/metro_case_summary.csv')

    # Extract needed outputs
    sim = SimulationRun.from_ocl_model(
        om=flm, chg_plan=flm.chg_schedule, site_caps=flm.num_chargers)
    print('\nEVALUATION RESULTS')
    sim.run_sim()
    sim.process_results()
    sim.print_results()

    # flm.plot_chg_ratios()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Define inputs
    depot_coords = (47.495809, -122.286190)
    # CSV file giving candidate charger sites
    # site_fname = '../data/all_king_cty_sites.csv'
    site_fname = '../data/king_cty_sites_trc.csv'

    beb_routes = [
        22, 116, 153, 154, 156, 157, 158, 159, 168, 169, 177, 179, 180,
        181, 182, 183, 186, 187, 190, 192, 193, 197
    ] + [
        101, 102, 111, 116, 143, 150, 157, 158, 159, 177, 178, 179, 180, 190,
        192, 193, 197
    ]
    beb_routes = list(set([str(r) for r in beb_routes]))

    # Battery capacity in kWh
    battery_cap = 525 * 0.8
    # Energy consumption rate in kWh per mile
    kwh_per_mile = 3
    # Power output of each charger
    chg_pwrs = 500 / 60
    # Charger construction cost
    s_cost = 200000
    c_cost = 698447

    n_max = 4
    alpha = 190 * 365 * 12 / 60

    flm = run_facility_location(
        route_list=beb_routes, site_file=site_fname, battery_cap=battery_cap,
        kwh_per_mile=kwh_per_mile, charge_power=chg_pwrs, site_cost=s_cost,
        charger_cost=c_cost, alpha=alpha, max_chargers=n_max, opt_gap=0.01,
        depot_coords=depot_coords,
        save_fname='../../results/metro_case_results.csv',
        summary_fname='../../results/metro_case_summary.csv',
        gtfs_dir='../data/gtfs/metro_mar24',
        test_date=datetime.datetime(2024, 3, 28)
    )


