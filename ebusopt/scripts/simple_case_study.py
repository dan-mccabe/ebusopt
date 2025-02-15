from ebusopt.opt.charger_location import ChargerLocationModel
from ebusopt.opt.benders_charge_scheduling import \
    solve_full_problem_gurobi, solve_with_benders, build_subproblem
from ebusopt.opt.heuristic_charge_scheduling import repeat_heuristic
from gurobipy import GRB
import logging
from datetime import datetime, timedelta
from numpy import sqrt
import pickle
from pyomo.environ import SolverFactory, value
import pyomo.environ as pyo
import time
import copy
import numpy as np


def build_flex_model(
        trips, delta, rho, sigma, tau, max_chg, u0, relax=False,
        delay_cutoff=60):
    """
    Deprecated precursor to charge scheduling optimization.
    """
    m = pyo.ConcreteModel()

    m.bus_trips = pyo.Set(initialize=trips, dimen=2)
    # Predecessor pairs require some processing
    dummy_trip = (-1, -1)
    dummy_arcs_start = [(*dummy_trip, i, j) for (i, j) in m.bus_trips]
    dummy_arcs_end = [(i, j, *dummy_trip) for (i, j) in m.bus_trips]

    diff_block_trips = [
        (i, j, i2, j2) for (i, j) in trips for (i2, j2) in trips
        if i != i2
    ]

    same_block_trips = [
        (i, j, i2, j2) for (i, j) in trips for (i2, j2) in trips
        if i == i2 and j2 > j
    ]

    arcs = diff_block_trips + same_block_trips \
        + dummy_arcs_start + dummy_arcs_end

    opt_arcs = list()
    for (i2, j2, i, j) in arcs:
        if (i2, j2) != dummy_trip and (i, j) != dummy_trip:
            delay_min = max(
                0, sigma[i, j] - tau[i2, j2] - sigma[i2, j2])
            if delay_min <= delay_cutoff:
                opt_arcs.append((i2, j2, i, j))
        else:
            opt_arcs.append((i2, j2, i, j))

    m.arcs = pyo.Set(initialize=opt_arcs, dimen=4)
    m.x = pyo.Var(m.arcs, within=pyo.Binary)
    m.u = pyo.Var(m.bus_trips, bounds=(0, max_chg))
    m.d = pyo.Var(m.bus_trips, within=pyo.NonNegativeReals)
    m.t = pyo.Var(m.bus_trips, within=pyo.NonNegativeReals)
    m.p = pyo.Var(m.bus_trips, within=pyo.NonNegativeReals)

    def obj_fn(model):
        return sum(model.d[i, j] for (i, j) in model.bus_trips)
    m.obj = pyo.Objective(rule=obj_fn)

    def delay_rule(model, i, j):
        if j == 0:
            return pyo.Constraint.Skip
        else:
            return model.d[i, j] >= \
                model.p[i, j-1] + model.t[i, j-1] - sigma[i, j]
    m.delay_constr = pyo.Constraint(m.bus_trips, rule=delay_rule)

    def plugin_time_1(model, i, j):
        return model.p[i, j] >= sigma[i, j] + model.d[i, j] + tau[i, j]
    m.finish_constr_1 = pyo.Constraint(m.bus_trips, rule=plugin_time_1)

    d_ub = 10
    t_ub = 60
    M1 = {
        (i2, j2, i, j): sigma[i2, j2] + tau[i2, j2] + d_ub + t_ub
        - sigma[i, j] - tau[i, j] for (i2, j2, i, j) in arcs
        if (i, j) != dummy_trip and (i2, j2) != dummy_trip}
    M2 = t_ub

    def plugin_time_2(model, i2, j2, i, j):
        if (i, j) == dummy_trip or (i2, j2) == dummy_trip:
            return pyo.Constraint.Skip

        return model.p[i, j] >= model.p[i2, j2] + model.t[i2, j2] - M1[
            i2, j2, i, j] * (1 - model.x[i2, j2, i, j])
    if not relax:
        m.finish_constr_2 = pyo.Constraint(m.arcs, rule=plugin_time_2)

    def chg_time_bd(model, i, j):
        other_trips = [dummy_trip] + [
            (i2, j2) for (i2, j2) in model.bus_trips if (i2, j2, i, j)
            in model.arcs]

        return model.t[i, j] <= M2 * sum(
            model.x[i2, j2, i, j] for (i2, j2) in other_trips)
    m.chg_time_constr = pyo.Constraint(m.bus_trips, rule=chg_time_bd)

    def connectivity(model, i, j):
        in_trips = [dummy_trip] + [
            (i2, j2) for (i2, j2) in model.bus_trips
            if (i2, j2, i, j) in model.arcs]

        out_trips = [dummy_trip] + [
            (i2, j2) for (i2, j2) in model.bus_trips
            if (i, j, i2, j2) in model.arcs]

        return sum(model.x[i2, j2, i, j] for (i2, j2) in in_trips) - sum(
            model.x[i, j, i2, j2] for (i2, j2) in out_trips) == 0
    m.connect_constr = pyo.Constraint(m.bus_trips, rule=connectivity)

    def one_arc_in(model, i, j):
        other_trips = [dummy_trip] + [
            (i2, j2) for (i2, j2) in model.bus_trips
            if (i2, j2, i, j) in model.arcs]
        return sum(model.x[i2, j2, i, j] for (i2, j2) in other_trips) <= 1
    m.one_arc_constr = pyo.Constraint(m.bus_trips, rule=one_arc_in)

    # Dummy node constraints: leave and return to depot
    m.leave_depot_constr = pyo.Constraint(
        expr=sum(m.x[a] for a in dummy_arcs_start) == 1)
    m.return_depot_constr = pyo.Constraint(
        expr=sum(m.x[a] for a in dummy_arcs_end) == 1)

    def track_constr(model, i, j):
        # No charge for trip j+1
        if (i, j+1) not in model.bus_trips:
            return pyo.Constraint.Skip

        # Handle initial charge
        if j == 0:
            last_chg = u0
        else:
            last_chg = model.u[i, j]

        return model.u[i, j+1] <= last_chg - delta[i, j] + rho * model.t[i, j]

    m.chg_track_constr = pyo.Constraint(m.bus_trips, rule=track_constr)

    def chg_lb(model, i, j):
        return delta[i, j] <= model.u[i, j]

    m.chg_bound_constr = pyo.Constraint(m.bus_trips, rule=chg_lb)

    return m


def build_simple_case(route_recov, service_hrs=6):
    # Build up example
    # Route information
    routes = ['A', 'B', 'C']
    # Distance in miles
    route_dist = {'A': 15, 'B': 25, 'C': 15}
    # Double distance and halve number of trips to test easier instance
    # (edited from original)
    route_dist = {k: route_dist[k]*2 for k in route_dist}
    # Time to complete in minutes
    route_time = {'A': 40, 'B': 90, 'C': 45}
    route_time = {r: timedelta(minutes=route_time[r]) for r in route_time}
    # Recovery time in minutes
    route_recov = {r: timedelta(minutes=route_recov[r]) for r in route_recov}
    # Headway in minutes
    route_hw = {'A': 30, 'B': 30, 'C': 60}
    route_hw = {r: timedelta(minutes=route_hw[r]) for r in route_hw}
    # Terminals of route
    route_terms = {'B': ('W', 'E'), 'A': ('S', 'N'), 'C': ('S', 'E')}
    # Create candidate charging sites and associated data
    all_locs = ['W', 'NW', 'N', 'NE', 'E', 'S']
    chg_sites = ['W', 'NW', 'S']
    chg_coords = {'W': (-12.5, 0), 'N': (0, 7.5), 'E': (12.5, 0), 'S': (0, -7.5),
                  'NW': (-10, 1), 'NE': (10, 1)}
    # Calculate charging distance (Euclidean)
    chg_dists = {(a, b): sqrt((chg_coords[a][0] - chg_coords[b][0])**2
                              + (chg_coords[a][1] - chg_coords[b][1])**2)
                 for a in all_locs for b in all_locs}
    coord_dists = {(chg_coords[a], chg_coords[b]):
                       sqrt((chg_coords[a][0] - chg_coords[b][0]) ** 2
                            + (chg_coords[a][1] - chg_coords[b][1]) ** 2)
                   for (a, b) in chg_dists}
    dh_data = {
        k: {'distance': coord_dists[k],
            'duration': coord_dists[k] * 60/25}
        for k in coord_dists}
    # Save the deadhead distances for use elsewhere
    with open('../data/processed/simple_case_deadhead.pickle', 'wb') as f:
        pickle.dump(dh_data, f)

    # Start time for first block on each route
    start_dt = datetime(2021, 8, 25, 7)
    # End time for last block on each route
    # (edited from original)
    end_dt = datetime(2021, 8, 25, 7 + service_hrs)

    # Create blocks, trips, and associated data
    blocks = list()
    trip_start_times = dict()
    trip_end_times = dict()
    trip_dists = dict()
    end_chg_dists = dict()
    start_chg_dists = dict()
    for r in routes:
        block_idx = 0
        new_block_start = start_dt
        # Create new routes until
        while new_block_start < start_dt + route_time[r] + route_recov[r]:
            block_idx += 1
            v_f = r + 'F' + str(block_idx)
            v_r = r + 'R' + str(block_idx)
            blocks.append(v_f)
            blocks.append(v_r)
            # Create trips for this block
            t_idx = 0
            # Add in trips to/from depot at end/start of day. These can just
            # be set to zero for the simple case, but we need a 0 index for
            # the model to work properly.
            trip_dists[v_f, 0] = 0
            trip_dists[v_r, 0] = 0
            block_start_min = (new_block_start - start_dt).total_seconds() / 60
            trip_start_times[v_f, 0] = block_start_min
            trip_start_times[v_r, 0] = block_start_min
            trip_end_times[v_f, 0] = block_start_min
            trip_end_times[v_r, 0] = block_start_min
            for s in chg_sites:
                end_chg_dists[v_f, 0, s] = 100
                end_chg_dists[v_r, 0, s] = 100
                start_chg_dists[v_f, 1, s] = 100
                start_chg_dists[v_r, 1, s] = 100

            # end_idx = t_idx + 1
            # trip_dists[v, end_idx] = gmap_last_trip['distance']
            # # Maintain penalty for late arrival to depot
            # trip_start_times[v, end_idx] = trip_end_times[v, end_idx - 1]
            # trip_end_times[v, end_idx] = trip_end_times[v, end_idx - 1]

            trip_start = new_block_start
            while trip_start < end_dt:
                t_idx += 1
                # Set start and end times
                for v in [v_f, v_r]:
                    trip_start_times[v, t_idx] = \
                        (trip_start - start_dt).total_seconds() / 60
                    trip_end_times[v, t_idx] = \
                        (trip_start + route_time[r] - start_dt).total_seconds() / 60
                    # Set distance
                    trip_dists[v, t_idx] = route_dist[r]

                # Set location of current trip end, next trip start
                end_loc_f = route_terms[r][t_idx % 2]
                end_loc_r = route_terms[r][(t_idx - 1) % 2]
                for s in chg_sites:
                    end_chg_dists[v_f, t_idx, s] = chg_dists[end_loc_f, s]
                    end_chg_dists[v_r, t_idx, s] = chg_dists[end_loc_r, s]
                    start_chg_dists[v_f, t_idx+1, s] = chg_dists[s, end_loc_f]
                    start_chg_dists[v_r, t_idx+1, s] = chg_dists[s, end_loc_r]
                # Set next trip start time
                trip_start = trip_start + route_time[r] + route_recov[r]
            # Create a new block by adding the headway to the first start
            new_block_start += route_hw[r]
    logging.info('Blocks: {}'.format(blocks))
    logging.info('Number of blocks: {}'.format(len(blocks)))
    veh_trip_pairs = list(trip_start_times.keys())
    # Assume no deadhead between trips
    inter_trip_times = {vt: 0 for vt in veh_trip_pairs}
    inter_trip_dists = {vt: 0 for vt in veh_trip_pairs}
    # Assume all trips take 3 kWh/mile
    energy_rates = {vt: 3 for vt in veh_trip_pairs}
    # Assume 25 mph to drive to/from chargers
    end_chg_times = {vts: end_chg_dists[vts] * 60 / 25 for vts in end_chg_dists}
    start_chg_times = {
        vts: start_chg_dists[vts] * 60 / 25 for vts in start_chg_dists}

    # Set site parameters
    good_sites = ['N', 'E', 'S', 'W']
    site_costs = {s: 5e5 if s in good_sites else 5e4 for s in chg_sites}
    # site_costs = {s: 400 if s in good_sites else 200 for s in chg_sites}
    charger_costs = {s: 698447 for s in chg_sites}
    max_ch = {s: 4 if s in good_sites else 4 for s in chg_sites}
    # max_ch = {s: 20 for s in chg_sites}
    chg_power = {s: 300/60 for s in chg_sites}

    # Set charge limits
    ch_lims = {v: 400 for v in blocks}

    opt_kwargs = dict(
        vehicles=blocks, veh_trip_pairs=veh_trip_pairs, chg_sites=chg_sites,
        chg_lims=ch_lims, trip_start_times=trip_start_times, max_chargers=max_ch,
        trip_end_times=trip_end_times, trip_dists=trip_dists,
        inter_trip_dists=inter_trip_dists, inter_trip_times=inter_trip_times,
        trip_start_chg_dists=start_chg_dists, trip_end_chg_dists=end_chg_dists,
        chg_rates=chg_power, site_costs=site_costs, charger_costs=charger_costs,
        trip_start_chg_times=start_chg_times, trip_end_chg_times=end_chg_times,
        energy_rates=energy_rates, zero_time=start_dt,
        depot_coords=chg_coords['S'])

    return opt_kwargs


def get_recharge_opt_params(rho_kw, service_hrs=6):
    """
    Set up simple case so it is suitable for one of our recharge
    planning algorithms.

    :param rho_kw: dict of charger power outputs in kW
    :param service_hrs: number of hours each BEB should be in service
        for consecutively
    :return: dict of instance parameters
    """
    rho = rho_kw / 60
    u_max = 400
    ocl_dict = build_simple_case(
        route_recov={'A': 20, 'B': 30, 'C': 60},
        service_hrs=service_hrs
    )

    # Exclude route B
    buses = [v for v in ocl_dict['vehicles'] if v[0] != 'B']
    # Filter down trips
    trips = [(v, t) for (v, t) in ocl_dict['veh_trip_pairs'] if v in buses]
    print('Number of buses: {}'.format(len(buses)))
    print('Number of trips: {}'.format(len(trips)))

    delta = {
        (v, t): ocl_dict['energy_rates'][v, t] * ocl_dict[
            'trip_dists'][v, t]
        for (v, t) in trips
    }
    sigma = {
        (v, t): ocl_dict['trip_start_times'][v, t] for (v, t) in trips
    }
    max_chg_time = dict()
    tau = {(v, t): ocl_dict['trip_end_times'][v, t]
                   - ocl_dict['trip_start_times'][v, t]
           for (v, t) in trips}
    for v in buses:
        t_v = sorted(tt for (vv, tt) in trips if vv == v)
        if v[1] == 'F':
            # Forward block starts at Charger 1
            for ix in range(int(len(t_v))):
                if ix % 2 == 0:
                    max_chg_time['Charger 1', v, ix] = 0
                    max_chg_time['Charger 2', v, ix] = u_max / rho
                else:
                    # Charging is available at charger 2, but not 1
                    max_chg_time['Charger 2', v, ix] = 0
                    max_chg_time['Charger 1', v, ix] = u_max / rho

        else:
            # Reverse block starts at Charger 2
            for ix in range(int(len(t_v))):
                if ix % 2 == 1:
                    max_chg_time['Charger 1', v, ix] = 0
                    max_chg_time['Charger 2', v, ix] = u_max / rho
                else:
                    max_chg_time['Charger 2', v, ix] = 0
                    max_chg_time['Charger 1', v, ix] = u_max / rho

    return {
        'trips': trips,
        'delta': delta,
        'sigma': sigma,
        'tau': tau,
        'rho': {'Charger 1': rho, 'Charger 2': rho},
        'max_chg_time': max_chg_time,
        'u_max': u_max
    }


def run_simple_case_ocl(route_recov):
    opt_kwargs = build_simple_case(route_recov)

    with open('simple_case.pickle', 'wb') as f:
        pickle.dump(opt_kwargs, f)

    flm = ChargerLocationModel(**opt_kwargs)
    flm.solve(alpha=190*365*12/60, simple_case=True, bu_kwh=400)
    flm.log_results()
    # print('Capital cost:', flm.capital_costs)
    flm.plot_chargers('../results/simple_util.pdf')


def run_flex_model():
    """
    Modify the original case study data and use it to test new model.
    """
    ocl_dict = build_simple_case({'A': 20, 'B': 30, 'C': 15})
    buses = ocl_dict['vehicles']
    buses = [v for v in buses if v[0] != 'B']

    trips = ocl_dict['veh_trip_pairs']
    # Exclude route B
    trips = [(v, t) for (v, t) in trips if v in buses]
    logging.info('Number of trips: {}'.format(len(trips)))

    # Update data for all trips to reflect charging can only happen when
    # bus is at the charging site
    all_trips = True
    if all_trips:
        delta = {
            (v, t): ocl_dict['energy_rates'][v, t] * ocl_dict[
                'trip_dists'][v, t]
            for (v, t) in ocl_dict['energy_rates']
        }
        sigma = ocl_dict['trip_start_times']
        max_chg_time = dict()
        tau = {(v, t): ocl_dict['trip_end_times'][v, t]
                       - ocl_dict['trip_start_times'][v, t]
               for (v, t) in ocl_dict['trip_start_times']}
        for v in buses:
            t_v = sorted(tt for (vv, tt) in trips if vv == v)
            if v[1] == 'F':
                # Forward block starts at charger
                for ix in range(int(len(t_v))):
                    if ix % 2 == 0:
                        max_chg_time[v, ix] = 0
                    else:
                        max_chg_time[v, ix] = 10000

            else:
                # Reverse block starts at opposite terminal from charger
                for ix in range(int(len(t_v))):
                    if ix % 2 == 0:
                        max_chg_time[v, ix] = 0
                    else:
                        max_chg_time[v, ix] = 10000
    else:
        delta = dict()
        sigma = dict()
        tau = dict()
        trips_reix = list()
        for v in buses:
            t_v = sorted(tt for (vv, tt) in trips if vv == v)
            if v[1] == 'F':
                # Forward trip starts at charger
                for ix in range(int(len(t_v) / 2)):
                    trips_reix.append((v, ix))
                    orig_ix = 2*ix+1

                    # Round trip start time
                    sigma[v, ix] = ocl_dict['trip_start_times'][v, orig_ix]

                    # Energy consumption
                    delta[v, ix] = ocl_dict['energy_rates'][v, orig_ix] * (
                            ocl_dict['trip_dists'][v, orig_ix]
                            + ocl_dict['trip_dists'][v, orig_ix + 1])

                    # Trip duration
                    tau[v, ix] = ocl_dict['trip_end_times'][v, orig_ix+1] \
                        - ocl_dict['trip_start_times'][v, orig_ix]

            else:
                # Reverse trip starts at opposite terminal from charger
                trips_reix.append((v, 0))
                sigma[v, 0] = ocl_dict['trip_start_times'][v, 1]
                delta[v, 0] = ocl_dict['energy_rates'][v, 1] * ocl_dict[
                    'trip_dists'][v, 1]
                tau[v, 0] = ocl_dict['trip_end_times'][v, 1] - ocl_dict[
                    'trip_start_times'][v, 1]

                for ix in range(1, int(len(t_v) / 2)):
                    trips_reix.append((v, ix))
                    orig_ix = 2*ix+1

                    # Round trip start time
                    sigma[v, ix] = ocl_dict['trip_start_times'][v, orig_ix]

                    if (v, orig_ix+1) in trips:
                        # Energy consumption
                        delta[v, ix] = ocl_dict['energy_rates'][v, orig_ix] * (
                                ocl_dict['trip_dists'][v, orig_ix]
                                + ocl_dict['trip_dists'][v, orig_ix+1])

                        # Trip duration is end of second trip minus start of
                        # first trip
                        tau[v, ix] = ocl_dict['trip_end_times'][v, orig_ix+1] \
                            - ocl_dict['trip_start_times'][v, orig_ix]

                    else:
                        delta[v, ix] = ocl_dict['energy_rates'][v, orig_ix] * (
                            ocl_dict['trip_dists'][v, orig_ix])
                        tau[v, ix] = ocl_dict['trip_end_times'][v, orig_ix] \
                                     - ocl_dict['trip_start_times'][v, orig_ix]

    rho = 300/60
    max_chg = 400
    u0 = 400
    t_0 = (-1, -1)

    logging.info('Number of buses in new case study: {}'.format(
        len(buses)))
    logging.info('Number of trips after simplifying: {}'.format(
        len(trips)))
    fm = build_flex_model(
        trips, delta, rho, sigma, tau, max_chg, u0, relax=False)
    logging.info('Total arcs: {}'.format(len(fm.arcs)))

    # Function used to add constraints
    d_ub = 10
    t_ub = 60
    M1 = {
        (i2, j2, i, j): sigma[i2, j2] + tau[i2, j2] + d_ub + t_ub
                        - sigma[i, j] - tau[i, j] for (i2, j2, i, j) in fm.arcs
        if (i, j) != t_0 and (i2, j2) != t_0}

    solver = SolverFactory('gurobi')

    lm_arcs = [a for a in fm.arcs if a[:2] != (-1, -1) and a[2:] != (-1, -1)]

    # Copy the model to make Lagrangian relaxation
    n_lg = 0
    lg_bd = [0] * n_lg
    for lg_ix in range(n_lg):
        lg_mult = 0.001 * np.random.rand(len(lm_arcs))
        lg_dict = {a: lg_mult[i] for i, a in enumerate(lm_arcs)}
        lm = copy.deepcopy(fm)
        lm.del_component(lm.obj)
        lm.obj = pyo.Objective(
            expr=sum(lm.d[i, j] for (i, j) in lm.bus_trips) + sum(
                lg_dict[i2, j2, i, j] * (
                        lm.p[i2, j2] + lm.t[i2, j2] - lm.p[i, j]
                        - M1[i2, j2, i, j] * (1 - lm.x[i2, j2, i, j]))
                for (i2, j2, i, j) in lm_arcs))
        # Solve Lagrangian model
        solver.solve(lm, tee=False)
        lg_bd[lg_ix] = value(lm.obj)

    if n_lg > 0:
        logging.info(
            'Best Lagrangian bound: {:.2f}'.format(max(lg_bd)))
        logging.info('\t{}'.format(lg_bd))

    # solver.options['TimeLimit'] = 20
    solve_start_time = time.time()
    solver.solve(fm, tee=True)
    logging.info('Initial solution time: {:.2f}'.format(
        time.time() - solve_start_time))

    def finish_time_2(model, i2, j2, i, j):
        if (i, j) == t_0 or (i2, j2) == t_0:
            return pyo.Constraint.Skip

        return model.p[i, j] >= model.p[i2, j2] + model.t[i2, j2] - M1[
            i2, j2, i, j] * (1 - model.x[i2, j2, i, j])

    infeas_flag = True
    viol_arcs = list()
    itr = 1
    while infeas_flag and itr <= 300:
        # Process most recent solution
        x_flex = [
            a for a in fm.arcs if value(fm.x[a]) > 0.99]

        # Separate and add constraints
        new_infeas = False
        for (i, j, i2, j2) in x_flex:
            if t_0 in [(i, j), (i2, j2)]:
                pass
            else:
                constr_tol = 0.01
                if value(fm.p[i2, j2]) + constr_tol < value(
                        fm.p[i, j]) + value(fm.t[i, j]):
                    logging.debug(
                        'Constraint violated: Trip {} ending at time {:.2f} is'
                        ' followed by trip {} with plugin time {:.2f}'.format(
                            (i, j), value(fm.p[i, j]) + value(fm.t[i2, j2]),
                            (i2, j2), value(fm.p[i2, j2])))

                    if (i, j, i2, j2) in viol_arcs:
                        logging.info(
                            'WARNING: we already generated a constraint for '
                            'indexes {}, but found another violation. Trip {} '
                            'ending at time {:.2f} is followed by trip {} with'
                            ' plugin time {:.2f}'.format(
                                (i, j, i2, j2), (i, j),
                                value(fm.p[i, j]) + value(fm.t[i2, j2]),
                                (i2, j2), value(fm.p[i2, j2])))
                    viol_arcs.append((i, j, i2, j2))
                    new_infeas = True

        if not new_infeas:
            logging.info('No new violated constraints detected.')
            infeas_flag = False

        else:
            logging.info(
                'Iteration {}: '
                'Incumbent model has {} queue time constraints.'.format(
                    itr, len(viol_arcs)))

            if itr > 1:
                fm.del_component(fm.finish_constr_2)
                fm.del_component(fm.finish_constr_2_index)

            # fm.add_component('viol_arcs', pyo.Set(initialize=set(viol_arcs)))
            if len(set(viol_arcs)) != len(viol_arcs):
                raise Warning(
                    'List of violated constraints include a constraint that '
                    'was already added to the model, indicating a modeling '
                    'error.')
            # Track how much weaker our bound gets from this constraint
            bound_dict = {
                (i2, j2, i, j): M1[i2, j2, i, j]
                for (i2, j2, i, j) in viol_arcs}
            logging.debug(
                'Big-M check: min {:.2f}, mean {:.2f}, max {:.2f}'.format(
                    min(bound_dict.values()),
                    sum(bound_dict.values()) / len(bound_dict.values()),
                    max(bound_dict.values())))

            # Add updated finish time constraints
            fm.add_component('finish_constr_2', pyo.Constraint(
                list(set(viol_arcs)), rule=finish_time_2))

            # Copy the model to make Lagrangian relaxation
            n_lg = 0
            lg_bd = [0]*n_lg
            for lg_ix in range(n_lg):
                lg_mult = 0.001*np.random.rand(len(viol_arcs))
                lg_dict = {a: lg_mult[i] for i, a in enumerate(viol_arcs)}
                lm = copy.deepcopy(fm)
                lm.del_component(lm.obj)
                lm.obj = pyo.Objective(
                    expr=sum(lm.d[i, j] for (i, j) in lm.bus_trips) + sum(
                        lg_dict[i2, j2, i, j] * (
                                lm.p[i2, j2] + lm.t[i2, j2] - lm.p[i, j]
                                - M1[i2, j2, i, j] * (1 - lm.x[i2, j2, i, j]))
                        for (i2, j2, i, j) in viol_arcs))
                # Solve Lagrangian model
                solver.solve(lm)
                lg_bd[lg_ix] = value(lm.obj)

            if n_lg > 0:
                logging.info(
                    'Best Lagrangian bound: {:.2f}'.format(max(lg_bd)))

            solver.solve(fm, warmstart=True)
            logging.info('Objective value: {:.2f}'.format(value(fm.obj)))
            itr += 1

    # Process result
    t_i = t_0
    x_sorted = [t_i]
    arcs_sorted = list()
    for ix in range(len(x_flex)):
        next_t = [(k, l) for (i, j, k, l) in x_flex if (i, j) == t_i]
        if not next_t:
            raise ValueError('Found no successor to {}'.format(t_i))
        elif next_t[0] == t_0:
            x_sorted.append(next_t[0])
            arcs_sorted.append((*t_i, *next_t[0]))
            break
        elif next_t[0] in x_sorted and next_t[0] != t_0:
            x_sorted.append(next_t[0])
            arcs_sorted.append((*t_i, *next_t[0]))
            break
        else:
            x_sorted.append(next_t[0])
            arcs_sorted.append((*t_i, *next_t[0]))
            t_i = next_t[0]

    logging.info('*** Final solution ***')
    logging.info('Total solution time: {:.2f}'.format(
        time.time() - solve_start_time))
    logging.info('Objective value: {:.2f}'.format(
        value(fm.obj)))
    logging.info('Charging sequence: {}'.format(x_sorted))
    logging.info(x_flex)
    # for (i2, j2, i, j) in x_flex:
    #     if (i2, j2) != (-1, -1) and (i, j) != (-1, -1):
    #         delay_a = max(
    #             0, sigma[i, j] - sigma[i2, j2] - tau[i2, j2])
    #         print('Delay for this arc:', delay_a)

    # print(len(x_sorted), len(x_flex))

    # Create a plot of the solution to validate


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    try_full = False
    max_delay = 1e12
    input_dict = get_recharge_opt_params(rho_kw=300, service_hrs=12)

    trips = input_dict['trips']
    sigma = input_dict['sigma']
    tau = input_dict['tau']
    delta = input_dict['delta']
    rho = input_dict['rho']
    max_chg_time = input_dict['max_chg_time']
    u_max = input_dict['u_max']
    chargers = ['Charger 1']

    # Try solving complete model
    if try_full:
        full_params = {'TimeLimit': 3600}

        opt_x, full_arcs = solve_full_problem_gurobi(
            trips=trips,
            chargers=chargers,
            sigma=sigma,
            tau=tau,
            delta=delta,
            chg_pwr=rho,
            max_chg_time=max_chg_time,
            max_chg=u_max,
            u0=u_max,
            delay_cutoff=max_delay,
            gurobi_params=full_params
        )

        chg_opps = [
            (c, i, j) for c in chargers for (i, j) in trips
            if max_chg_time[c, i, j] > 0
        ]
        # Which trips don't have charging?
        skip_trips = [
            k for k in chg_opps if k not in opt_x
        ]

        # Build the subproblem
        sp_m = build_subproblem(
            trips_skip=skip_trips,
            chargers=chargers,
            arcs_chg=full_arcs,
            trips=trips,
            delta=delta,
            chg_pwr=rho,
            sigma=sigma,
            tau=tau,
            max_chg_time=max_chg_time,
            max_chg=u_max,
            u0=u_max
        )
        sp_m.optimize()
        print('** Subproblem validation **')
        if sp_m.status == GRB.OPTIMAL:
            print('Subproblem objective value: {:.2f}'.format(sp_m.ObjVal))
        elif sp_m.status == GRB.INFEASIBLE:
            print('Subproblem was found to be infeasible!')
        else:
            raise ValueError(
                'Unrecognized solver status: {}'.format(sp_m.status))

    soln_dict = repeat_heuristic(
        case_data=input_dict,
        chargers=chargers,
        rho=rho,
        u_max=u_max,
        n_runs=500,
        random_mult=0.5
    )
    # soln_dict = None
    run_benders = False
    if run_benders:
        solve_with_benders(
            case_data=input_dict,
            chargers=chargers,
            rho=rho,
            u_max=u_max,
            heur_solns=soln_dict,
            cut_gap=0.5,
            delay_cutoff=min(soln_dict.values()),
            time_limit=3600
        )
