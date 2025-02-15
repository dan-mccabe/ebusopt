import numpy as np
import gurobipy as gp
import time
import logging
from gurobipy import GRB
from ebusopt.opt.benders_charge_scheduling import build_subproblem, to_df
logger = logging.getLogger('heuristics')

def solve_single_bus_model(
        trips, chargers, costs, sigma, tau, delta, chg_pwr, max_chg_time,
        max_chg, u0):
    m = gp.Model()
    chg_ix = [
        (c, i, j) for c in chargers for (i, j) in trips
        if max_chg_time[c, i, j] > 0
    ]

    d_vars = m.addVars(trips, obj=1, lb=0, name='d')
    m._d = d_vars
    u_vars = m.addVars(trips, lb=delta, ub=max_chg, name='u')
    t_vars = m.addVars(
        chg_ix, obj=costs, lb=0, ub=max_chg_time, name='t')
    m._t = t_vars

    for (i, j) in [t for t in trips if t[1] > 0]:
        c_feas = [
            cc for cc in chargers if (cc, i, j - 1) in chg_ix
        ]
        if not c_feas:
            m.addConstr(
                d_vars[i, j] >= sigma[i, j - 1] + d_vars[i, j - 1] + tau[
                    i, j - 1] - sigma[i, j]
            )
        for c in c_feas:
            m.addConstr(
                d_vars[i, j] >= sigma[i, j-1] + d_vars[i, j-1] + tau[i, j-1]
                + t_vars[c, i, j-1] - sigma[i, j]
            )

    # Initial charge values
    buses = list(set(t[0] for t in trips))
    for i in buses:
        m.addConstr(u_vars[i, 0] == u0)

    # Charge tracking
    for (i, j) in trips:
        # No charge for trip j+1
        if (i, j + 1) not in trips:
            continue

        # Handle initial charge
        if j == 0:
            last_chg = u0
        else:
            last_chg = u_vars[i, j]

        c_feas = [
            cc for cc in chargers if (cc, i, j) in chg_ix
        ]
        m.addConstr(
            u_vars[i, j + 1] <= last_chg - delta[i, j] + sum(
                chg_pwr[c] * t_vars[c, i, j] for c in c_feas
            )
        )

    m.Params.LogToConsole = 0
    m.optimize()

    return m


def run_two_stage_model(
        trips, chargers, delta, sigma, tau, rho, max_chg_time,
        u_max, random_mult=3.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    obj_val = 0
    heur_plugins = dict()
    skip_trips = list()
    infeas_vehs = list()
    all_buses = list(set(t[0] for t in trips))
    for v in all_buses:
        v_trips = [k for k in trips if k[0] == v]
        costs = {(c, i, j): random_mult*rng.normal(loc=-2, scale=1)
                 for c in chargers for (i, j) in v_trips}

        # If multiple chargers are close by, pick the one with the
        # minimum cost
        max_chg_zeros = list()
        for (i, j) in v_trips:
            c_avail = [c for c in chargers if max_chg_time[c, i, j] > 0]
            if len(c_avail) > 1:
                costs_avail = {c: costs[c, i, j] for c in c_avail}
                c_min = min(costs_avail, key=costs_avail.get)
                c_zero = [c for c in c_avail if c != c_min]
                max_chg_zeros += [(c, i, j) for c in c_zero]

        max_chg_adj = {
            (c, i, j): 0 if (c, i, j) in max_chg_zeros else max_chg_time[c, i, j]
            for (c, i, j) in max_chg_time if i == v
        }
        m = solve_single_bus_model(
            trips=v_trips,
            chargers=chargers,
            costs=costs,
            delta=delta,
            sigma=sigma,
            tau=tau,
            chg_pwr=rho,
            max_chg_time=max_chg_adj,
            max_chg=u_max,
            u0=u_max
        )

        if m.status == GRB.OPTIMAL:
            t_opt = m.getAttr('X', m._t)
        elif m.status == GRB.INFEASIBLE:
            logger.debug(
                f'Single-bus problem was found to be infeasible for block {v}'
            )
            # Mark this block as infeasible
            # TODO: would be nice to track these infeasible vehicles
            #   across iterations to save time when we run
            #   repeat_heuristic()
            infeas_vehs.append(v)
            # Move on to the next block
            continue
        else:
            raise ValueError(
                'Unrecognized solver status: {}'.format(m.status))

        t_nonzero = {k: v for k, v in t_opt.items() if v > 0.001}
        # Trip/charger pairs where optimal charging duration was zero
        skip_opt = [
            k for k in t_opt if t_opt[k] < 0.001
        ]
        skip_trips += skip_opt
        # Trip/charger pairs where charging was not possible (need to
        # add these separately since charging was made infeasible and
        # they never had variable indices created)
        skip_trips += max_chg_zeros

        d_opt = m.getAttr('X', m._d)
        d_opt = {k: v for k, v in d_opt.items()}
        obj_val += sum(d_opt.values())

        for c, i, j in t_nonzero:
            heur_plugins[c, i, j] = sigma[i, j] + tau[i, j] + d_opt[i, j]

    plugin_arcs = list()
    for c in chargers:
        c_plugins = {k: heur_plugins[k] for k in heur_plugins if k[0] == c}
        plugin_order = sorted(c_plugins, key=c_plugins.get)

        plugin_arcs += [
            (c,) + plugin_order[ix][1:] + plugin_order[ix+1][1:]
            for ix in range(len(plugin_order)-1)
        ]

    # Remove infeasible blocks from the analysis
    sp_trips = [
        t for t in trips if t[0] not in infeas_vehs
    ]

    n_chg_trips = 1e6
    n_trips_changed = True
    sp_skip = skip_trips
    plugins_revised = plugin_arcs
    itr = 1
    # The reason for placing the solution within a loop is that
    # sometimes after solving the subproblem, we find that the charging
    # time equals zero for some trips that were not set to be skipped
    # (and, in turn, were included in arcs_chg). So, we repeatedly
    # optimize until we get a solution where all arcs are used. This
    # ensures our objective is accurately calculating the total delay.
    # Otherwise, we tend to overestimate because we include charging
    # arcs that aren't actually used when calculating delay.
    while n_trips_changed:
        # Build and solve the subproblem
        sp_m = build_subproblem(
            trips_skip=sp_skip,
            chargers=chargers,
            arcs_chg=plugins_revised,
            trips=sp_trips,
            delta=delta,
            chg_pwr=rho,
            sigma=sigma,
            tau=tau,
            max_chg_time=max_chg_time,
            max_chg=u_max,
            u0=u_max,
            best_delay=1e100
        )
        sp_m.optimize()

        if sp_m.status == GRB.OPTIMAL:
            # Remove unused arcs and re-optimize
            t_opt = sp_m.getAttr('X', sp_m._t)
            t_nonzero = {k: v for k, v in t_opt.items() if v > 0.001}
            if len(t_nonzero) == n_chg_trips:
                n_trips_changed = False
            else:
                n_chg_trips = len(t_nonzero)
                logger.debug(
                    'Iteration {} subproblem objective: {:.2f},'
                    'charging after {} trips.'.format(
                    itr, sp_m.getObjective().getValue(), n_chg_trips
                    )
                )
                itr += 1
                sp_delay = sp_m.getAttr('X', sp_m._d)
                sp_skip = [k for k in t_opt if t_opt[k] < 0.001]

                sp_plugins = dict()
                for c, i, j in t_nonzero:
                    sp_plugins[c, i, j] = sigma[i, j] + tau[i, j] + sp_delay[i, j]

                plugins_revised = list()
                for c in chargers:
                    c_plugins = {
                        k: sp_plugins[k] for k in sp_plugins if k[0] == c
                    }
                    plugin_order = sorted(c_plugins, key=c_plugins.get)

                    plugins_revised += [
                        (c,) + plugin_order[ix][1:] + plugin_order[ix + 1][1:]
                        for ix in range(len(plugin_order) - 1)
                    ]

        elif sp_m.status == GRB.INFEASIBLE:
            raise ValueError('Subproblem was found to be infeasible!')
        else:
            raise ValueError(
                'Unrecognized solver status: {}'.format(sp_m.status))

    df = to_df(sp_m, sigma, tau)
    return {
        'plugin_arcs': plugin_arcs,
        'obj_val': sp_m.ObjVal,
        'df': df
    }


def repeat_heuristic(
        case_data, chargers, rho, u_max, n_runs, random_mult,
        return_type: str = 'dict', rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    # Run heuristic
    best_obj = 1e12
    results = np.zeros(n_runs)
    start_t = time.time()
    soln_dict = dict()
    for i in range(n_runs):
        result = run_two_stage_model(
            random_mult=random_mult,
            trips=list(case_data['sigma'].keys()),
            chargers=chargers,
            sigma=case_data['sigma'],
            tau=case_data['tau'],
            delta=case_data['delta'],
            rho=rho,
            max_chg_time=case_data['max_chg_time'],
            u_max=u_max,
            rng=rng
        )
        plugin_arcs = result['plugin_arcs']
        obj_val = result['obj_val']
        df = result['df']
        if obj_val < best_obj:
            logger.info(
                'Found new best solution with objective value {:.2f} at time '
                '{:.2f}'.format(obj_val, time.time() - start_t)
            )
            best_obj = obj_val

            # Write solution
            best_df = df
            best_df.to_csv('incumbent_soln.csv', index=False)

        # Keep track of unique solutions
        # Note that plugin_arcs is already sorted, so it's safe to
        # convert to tuple and use that to check if it's already been
        # found.
        if tuple(plugin_arcs) not in soln_dict:
            soln_dict[tuple(plugin_arcs)] = obj_val

        results[i] = obj_val
        if obj_val == 0:
            break
    run_t = time.time() - start_t

    logger.info('Performed {} heuristic runs in {:.2f} s'.format(i+1, run_t))
    logger.info('Obtained {} unique feasible solutions.'.format(len(soln_dict)))
    logger.info(
        'Best/mean/worst objective values: {:.1f}/{:.1f}/{:.1f}'.format(
            np.min(results),
            np.mean(results[:i + 1]),
            np.max(results)
        )
    )

    if return_type == 'dict':
        return soln_dict
    elif return_type == 'df':
        return best_df
    else:
        raise ValueError('return_type must be either "dict" or "df"')



