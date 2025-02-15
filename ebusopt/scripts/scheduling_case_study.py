from ebusopt.gtfs_beb import GTFSData
from datetime import datetime
import pandas as pd
from ebusopt.opt.heuristic_charge_scheduling import repeat_heuristic
from ebusopt.opt.benders_charge_scheduling import solve_with_benders
from ebusopt.scripts.script_helpers import build_trips_df, \
    build_scheduling_inputs, add_realtime_durations, load_realtime_summary
from numpy.random import default_rng
from ebusopt.vis import plot_trips_and_terminals
import logging


def set_up_sched_case(
        gtfs_dir, date_in, depot_coords, locs_df, routes, u_max, charger_kw,
        duration_quantile=None, kwh_per_mi=3., show_map=False
):
    if duration_quantile is None:
        duration_method = 'scheduled'
    else:
        duration_method = 'quantile'

    # Initialize GTFS
    gtfs = GTFSData.from_dir(dir_name=gtfs_dir)

    beb_trips = build_trips_df(
        gtfs=gtfs,
        date=date_in,
        routes=routes,
        routes_60=[],
        depot_coords=depot_coords,
        add_depot_dh=True,
        add_trip_dh=True
    )

    if show_map:
        inst_map = plot_trips_and_terminals(
            trips_df=beb_trips, locs_df=locs_df,
            shapes_df=gtfs.shapes_df)
        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'scale': 3
                # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        inst_map.show(config=config)

    beb_trips = add_realtime_durations(
        trips_to_lookup=beb_trips,
        realtime_summary=load_realtime_summary(),
        sim_all=False
    )
    logging.info(
        'There are {} total trips to be served by {} BEB blocks.'.format(
            len(beb_trips),
            beb_trips['block_id'].nunique()
        )
    )

    # beb_trips = predict_kwh_per_mi(beb_trips)
    beb_trips['kwh_per_mi'] = kwh_per_mi

    blocks_excl = [str(b) for b in [7157686, 7160963, 7157747, 7160999]]
    beb_trips = beb_trips[~beb_trips.isin(blocks_excl)].copy()

    beb_trips['kwh'] = beb_trips['kwh_per_mi'] * (
            beb_trips['total_dist'] + beb_trips['dh_dist']
    )
    total_kwh = beb_trips.groupby('block_id')['kwh'].sum()
    opp_chg_blocks = total_kwh[total_kwh > u_max].index.tolist()
    logging.info(
        '{} blocks must use opportunity charging; {} can use depot charging'
        ' only.'.format(len(opp_chg_blocks),
                       len(total_kwh[total_kwh <= u_max]))
    )
    logging.info(
        'Opportunity charging buses complete {} trips.'.format(
            len(beb_trips[beb_trips['block_id'].isin(opp_chg_blocks)])
        )
    )

    chargers_df = locs_df.rename(columns={'y': 'lat', 'x': 'lon'})
    chargers_df['kw'] = charger_kw

    return build_scheduling_inputs(
        beb_trips=beb_trips,
        chargers_df=chargers_df,
        u_max=u_max,
        energy_method='constant',
        constant_val=kwh_per_mi,
        duration_method=duration_method,
        duration_quantile=duration_quantile
    )


def run_multi_charger_case(
        u_max=300., rho_kw=250, n_runs=100, scenario='a', duration_quantile=None,
        show_map=False, random_mult=1., kwh_per_mi=3., try_full=False):
    # chargers = [
    #     'Burien Transit Center 1', 'Burien Transit Center 2',
    #     'Federal Way Transit Center',
    #     'Kent Des Moines Link Station',
    #     'Kent Transit Center 1', 'Kent Transit Center 2',
    #     'South Renton P&R'
    #     # 'Kent Station'  # , 'Renton Landing',
    # ]
    if scenario == 'a':
        locs_df = pd.read_csv('../data/sched_sites.csv', index_col=0)
    elif scenario == 'b':
        locs_df = pd.read_csv('../data/sched_sites_b.csv', index_col=0)
    else:
        raise ValueError('Scenario must be either \'a\' or \'b\'')
    chargers = locs_df.index.tolist()
    rho = {c: rho_kw / 60 for c in chargers}
    # routes = [
    #     101, 102, 105, 106, 107, 131, 132, 150, 153, 156, 160, 161, 165, 168,
    #     177, 182, 183, 187, 193, 240
    # ]
    routes = ['F Line', 'H Line', 131, 132, 150, 153, 161, 165]
    routes = [str(r) for r in routes]
    case_data = set_up_sched_case(
        gtfs_dir='../data/gtfs/metro_mar24',
        date_in=datetime(2024, 4, 3),
        depot_coords=(47.495809, -122.286190),
        locs_df=locs_df,
        routes=routes,
        u_max=u_max,
        charger_kw=rho_kw,
        show_map=show_map,
        kwh_per_mi=kwh_per_mi,
        duration_quantile=duration_quantile
    )
    return repeat_heuristic(
        case_data, chargers, rho, u_max, n_runs, random_mult, return_type='df',
        rng=default_rng(100)

    )
    # plot_charger_timelines(
    #     fname='incumbent_soln.csv',
    #     zero_time=datetime(2024, 4, 3)
    # )

    # if try_full:
    #     solve_full_problem_gurobi(
    #         trips=list(case_data['sigma'].keys()),
    #         sigma=case_data['sigma'],
    #         tau=case_data['tau'],
    #         delta=case_data['delta'],
    #         max_chg_time=case_data['max_chg_time'],
    #         chargers=chargers,
    #         chg_pwr=rho,
    #         max_chg=u_max,
    #         u0=u_max,
    #         gurobi_params={'TimeLimit': 3600}
    #     )


def run_benders(
        u_max=300, rho_kw=250, n_runs=100, add_delay=False, random_mult=1.):
    chargers = ['Burien Transit Center']
    rho = {c: rho_kw / 60 for c in chargers}
    routes = ['F Line', 'H Line']
    case_data = set_up_sched_case(
        chargers=chargers,
        routes=routes,
        u_max=u_max,
        rho=rho,
        add_delay=add_delay
    )

    soln_dict = repeat_heuristic(
        case_data=case_data,
        chargers=chargers,
        rho=rho,
        u_max=u_max,
        n_runs=n_runs,
        random_mult=random_mult
    )
    best_obj = min(soln_dict.values())

    if best_obj > 0:
        solve_with_benders(
            case_data=case_data,
            chargers=chargers,
            rho=rho,
            u_max=u_max,
            heur_solns=soln_dict,
            cut_gap=0.5
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # run_single_charger_case(rho_kw=350, add_delay=False, show_map=False)

    # 0.01: 225/300/639
    soln_df = run_multi_charger_case(
        rho_kw=220, show_map=True, random_mult=0.5, scenario='a',
        n_runs=500, u_max=525*0.75, kwh_per_mi=3.19, duration_quantile=None
    )

    soln_df.to_csv('sched_a_soln.csv')

    soln_df_b = run_multi_charger_case(
        rho_kw=220, show_map=False, random_mult=0.5, scenario='b',
        n_runs=500, u_max=525*0.75, kwh_per_mi=3.19, duration_quantile=None
    )

    soln_df_b.to_csv('sched_b_soln.csv')

    # df = pd.read_csv(
    #     'incumbent_soln.csv',
    #     dtype={'block_id': str}
    # )
    # plot_charger_timelines(df, datetime(2024, 4, 3),
    #                        True, False, 'Charger Timeline at {}')
    # run_benders(u_max=300, rho_kw=450, n_runs=100, add_delay=True)

