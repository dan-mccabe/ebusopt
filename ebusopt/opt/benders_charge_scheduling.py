import numpy as np
import logging
import pandas as pd
import gurobipy as gp
import itertools
import random
from gurobipy import GRB
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
logger = logging.getLogger('benders')

def sort_result(opt_arcs, dummy_trip=(-1, -1)):
    """
    Sort list of arcs obtained from solution to get a complete path.

    :param dummy_trip: dummy trip, typically (-1, -1)
    :param opt_arcs: list of arcs in solution as 5-tuples, e.g.
        ('Charger 1', 'A', 1, 'B', 2)
    :return: opt_arcs sorted as a path/tour
    """
    chargers = list(set(a[0] for a in opt_arcs))
    ordered_trips = dict()
    for c in chargers:
        t_i = dummy_trip
        x_sorted = [t_i]
        arcs_sorted = list()
        for ix in range(len(opt_arcs)):
            next_t = [(k, l) for (chg, i, j, k, l) in opt_arcs
                      if chg == c and (i, j) == t_i]
            if not next_t:
                raise ValueError('Found no successor to {}'.format(t_i))
            elif next_t[0] == dummy_trip:
                x_sorted.append(next_t[0])
                arcs_sorted.append((*t_i, *next_t[0]))
                break
            elif next_t[0] in x_sorted and next_t[0] != dummy_trip:
                x_sorted.append(next_t[0])
                arcs_sorted.append((*t_i, *next_t[0]))
                break
            else:
                x_sorted.append(next_t[0])
                arcs_sorted.append((*t_i, *next_t[0]))
                t_i = next_t[0]

        ordered_trips[c] = x_sorted

    return ordered_trips


def gen_arc_costs(
        feas_trips, chargers, start_times, trip_times,
        dummy_trip=(-1, -1), delay_cutoff=1e6, fully_directed=False):
    """
    Generate costs associated with each charging arc, to be used in the
    MP objective function. Also return a list of all the arcs to be
    included, where any arcs that force a delay of delay_cutoff or more
    will be omitted.

    :param feas_trips: list of trips for which charging is feasible
    :param chargers: list of chargers
    :param start_times: dict of start times of all trips
    :param trip_times: dict of passenger trip durations of all trips
    :param dummy_trip: tuple used to represent dummy start/end node
    :param delay_cutoff: delay limit for including arcs in network
    :param fully_directed: True if the network should be fully directed
        (which may be suboptimal, but probably not), False otherwise
    :return:
      - opt_arcs, list of arcs to be included in network
      - costs, dict of arc costs
    """
    dummy_arcs_start = [
        (c, dummy_trip[0], dummy_trip[1], i, j) for c in chargers for (i, j) in feas_trips[c]
    ]

    dummy_arcs_end = [
        (c, i, j, dummy_trip[0], dummy_trip[1]) for c in chargers for (i, j) in feas_trips[c]
    ]

    if fully_directed:
        real_arcs = list()
        for c in chargers:
            for (i, j), (i2, j2) in itertools.combinations(feas_trips[c], 2):
                # Add exactly one directed arc between these trips
                if start_times[i, j] + trip_times[i, j] <= start_times[
                  i2, j2] + trip_times[i2, j2]:
                    real_arcs.append((c, i, j, i2, j2))

                else:
                    real_arcs.append((c, i2, j2, i, j))

        opt_arcs = real_arcs + dummy_arcs_start + dummy_arcs_end
        costs = {a: 1 for a in opt_arcs}

    else:
        same_block_trips = [
            (c, i, j, i2, j2) for c in chargers for (i, j) in feas_trips[c]
            for (i2, j2) in feas_trips[c] if i == i2 and j2 > j
        ]

        diff_block_trips = [
            (c, i, j, i2, j2) for c in chargers for (i, j) in feas_trips[c]
            for (i2, j2) in feas_trips[c] if i != i2
        ]
        arcs = diff_block_trips + same_block_trips \
               + dummy_arcs_start + dummy_arcs_end

        opt_arcs = list()
        costs = dict()
        for (c, i2, j2, i, j) in arcs:
            if (i2, j2) == dummy_trip or (i, j) == dummy_trip:
                costs[c, i2, j2, i, j] = 0
                opt_arcs.append((c, i2, j2, i, j))

            else:
                # Lower bound on delay if we take this arc
                if (i, j+1) in trip_times:
                    # Difference between trip end time of origin node and
                    # next trip start time of destination node
                    delay_lb = start_times[i2, j2] + trip_times[i2, j2] - (
                            start_times[i, j+1]
                    )

                    # Only add arcs that fall below our delay cutoff
                    if delay_lb <= delay_cutoff:
                        costs[c, i2, j2, i, j] = max(0, delay_lb)
                        opt_arcs.append((c, i2, j2, i, j))

                else:
                    costs[c, i2, j2, i, j] = 0
                    opt_arcs.append((c, i2, j2, i, j))

    return opt_arcs, costs


def solve_full_problem_gurobi(
        trips, chargers, sigma, tau, delta, chg_pwr, max_chg_time, max_chg, u0,
        delay_cutoff=1e6, gurobi_params=None):
    """
    Solve complete charge scheduling problem directly with Gurobipy.

    :param trips: list of all trips
    :param chargers: list of all chargers
    :param sigma: dict of start times of all trips
    :param tau: dict of duration of all trips
    :param delta: dict of energy consumption of all trips
    :param chg_pwr: dict of charger power (in kWh/min) of all chargers
    :param max_chg_time: dict of maximum feasible charging time for all
        combinations of chargers and trips
    :param max_chg: float of usable battery capacity for all buses
    :param u0: float of initial charge for all buses
    :param delay_cutoff: delay cutoff used in gen_arc_costs
    :param gurobi_params: dict of additional options supplied to Gurobi
        solver
    :return: trips and arcs used in optimal solution, also prints results
    """
    dummy_trip = (-1, -1)
    m = gp.Model()

    # Arcs require some processing
    feas_trip_chg_pairs = [
        (c, i, j) for c in chargers for i, j in trips
        if max_chg_time[c, i, j] > 0
    ]

    feas_trips_by_charger = {
        c: [(i, j) for cc, i, j in feas_trip_chg_pairs if cc == c]
        for c in chargers
    }

    opt_arcs, _ = gen_arc_costs(
        feas_trips=feas_trips_by_charger,
        chargers=chargers,
        start_times=sigma,
        trip_times=tau,
        delay_cutoff=delay_cutoff,
        dummy_trip=dummy_trip
    )

    y_vars = m.addVars(opt_arcs, vtype=GRB.BINARY, name='y')
    x_vars = m.addVars(feas_trip_chg_pairs, vtype=GRB.BINARY, name='x')
    u_vars = m.addVars(trips, lb=delta, ub=max_chg, name='u')
    d_vars = m.addVars(trips, obj=1, lb=0, name='d')
    t_vars = m.addVars(feas_trip_chg_pairs, lb=0, name='t')
    p_vars = m.addVars(trips, lb=0, name='p')

    # Connectivity constraints
    for (c, i, j) in feas_trip_chg_pairs:
            in_trips = [dummy_trip] + [
                (i2, j2) for (i2, j2) in trips
                if (c, i2, j2, i, j) in opt_arcs]

            out_trips = [dummy_trip] + [
                (i2, j2) for (i2, j2) in trips
                if (c, i, j, i2, j2) in opt_arcs]

            m.addConstr(
                sum(y_vars[c, i2, j2, i, j] for (i2, j2) in in_trips) - sum(
                    y_vars[c, i, j, i2, j2] for (i2, j2) in out_trips) == 0
            )

    # Dummy node constraints: leave and return to depot
    for c in chargers:
        dummy_arcs_start = [
            (c, *dummy_trip, i, j) for (i, j) in feas_trips_by_charger[c]
        ]
        dummy_arcs_end = [
            (c, i, j, *dummy_trip) for (i, j) in feas_trips_by_charger[c]
        ]

        m.addConstr(
            sum(y_vars[a] for a in dummy_arcs_start) == 1)
        m.addConstr(
            sum(y_vars[a] for a in dummy_arcs_end) == 1)

    # Relationship between x and y variables
    for (c, i, j) in feas_trip_chg_pairs:
        in_trips = [dummy_trip] + [
            (i2, j2) for (i2, j2) in trips
            if (c, i2, j2, i, j) in opt_arcs]
        m.addConstr(
            x_vars[c, i, j] == sum(
                y_vars[c, i2, j2, i, j] for (i2, j2) in in_trips)
        )

    # Delay setting
    for (i, j) in [t for t in trips if t[1] > 0]:
        c_feas = [
          cc for cc in chargers if (i, j-1) in feas_trips_by_charger[cc]
        ]
        if not c_feas:
            m.addConstr(
                d_vars[i, j] >= p_vars[i, j - 1] - sigma[i, j])
        for c in c_feas:
            m.addConstr(
                d_vars[i, j] >= p_vars[i, j-1] + t_vars[c, i, j-1] - sigma[i, j])

    # Plugin time universal
    for (i, j) in trips:
        m.addConstr(p_vars[i, j] >= sigma[i, j] + d_vars[i, j] + tau[i, j])

    # Plugin time conditional (big-M)
    big_m = 1e5
    for (c, i2, j2, i, j) in opt_arcs:
        if (i, j) == dummy_trip or (i2, j2) == dummy_trip:
            continue

        else:
            m.addConstr(
                p_vars[i, j] >= p_vars[i2, j2] + t_vars[c, i2, j2]
                - big_m * (1 - y_vars[c, i2, j2, i, j])
            )

    # Constraint relating x and t
    for (c, i, j) in feas_trip_chg_pairs:
        m.addConstr(
            t_vars[c, i, j] <= max_chg_time[c, i, j] * x_vars[c, i, j]
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
            cc for cc in chargers if (i, j) in feas_trips_by_charger[cc]
        ]
        m.addConstr(
            u_vars[i, j + 1] <= last_chg - delta[i, j] + sum(
                chg_pwr[c] * t_vars[c, i, j] for c in c_feas
            )
        )

    if gurobi_params is not None:
        if not isinstance(gurobi_params, dict):
            raise ValueError('gurobi_params must be a dict')

        for k, v in gurobi_params.items():
            setattr(m.Params, k, v)

    m.optimize()

    # Extract results from solved model
    y_opt = {k: v.X for k, v in y_vars.items()}
    t_opt = {k: v.X for k, v in t_vars.items()}

    t_vals_nonzero = {t: t_opt[t] for t in t_opt.keys() if t_opt[t] > 0.01}

    print('\n **** Complete Model Results ****')
    print('Objective value: {:.2f}'.format(m.ObjVal))
    print(
        'Charging durations: {}'.format(
            t_vals_nonzero
        )
    )

    used_arcs = [a for a in opt_arcs if y_opt[a] > 0.99]
    print(
        'Charging arcs used: {}'.format(
            sort_result(used_arcs)
        )
    )

    chg_trips = list(t_vals_nonzero.keys())

    return chg_trips, used_arcs


def build_subproblem(
        chargers, trips_skip, arcs_chg, trips, delta, chg_pwr, sigma, tau,
        max_chg_time, max_chg, u0, best_delay=1e6):
    """
    Build an instance of the Benders subproblem with Guronipy.

    :param chargers: list of available chargers
    :param trips_skip: list of opportunities where charging has been
        skipped in MP solution (i.e., x^l_i = 0 in the master problem)
    :param arcs_chg: list of charging arcs used in MP solution (i.e.,
        y^l_ij = 1 in the MP)
    :param trips: list of all trips
    :param delta: dict of energy consumption per trip
    :param chg_pwr: dict of power output for each charger in kWh/min
    :param sigma: dict of start times for all trips
    :param tau: dict of trip durations for all trips
    :param max_chg_time: dict of max charge time for all trip-charger
        pairs
    :param max_chg: usable battery capacity of all buses in kWh
    :param u0: initial charge of all buses in kWh
    :param best_delay: best objective values from incumbent solution,
        in minutes of total delay
    :return: constructed (not solved) Gurobi LP model of SP
    """
    dummy_trip = (-1, -1)
    chg_ix = [
        (c, i, j) for c in chargers for (i, j) in trips
        if max_chg_time[c, i, j] > 0
    ]

    m = gp.Model()
    m.Params.LogToConsole = 0
    u_vars = m.addVars(trips, lb=delta, ub=max_chg, name='u')
    m._u = u_vars
    d_vars = m.addVars(trips, obj=1, lb=0, name='d')
    m._d = d_vars
    t_vars = m.addVars(chg_ix, lb=0, ub=max_chg_time, name='t')
    m._t = t_vars
    p_vars = m.addVars(trips, lb=0, name='p')
    m._p = p_vars
    m._chargers = chargers
    m._trips = trips

    # Delay setting
    for (i, j) in [t for t in trips if t[1] > 0]:
        c_feas = [
            cc for cc in chargers
            if (cc, i, j - 1) in chg_ix
        ]
        if not c_feas:
            m.addConstr(
                d_vars[i, j] >= p_vars[i, j - 1] - sigma[i, j]
            )
        for c in c_feas:
            m.addConstr(
                d_vars[i, j] >= p_vars[i, j - 1] + t_vars[c, i, j - 1] - sigma[
                    i, j],
                name='delay_{}_{}_{}'.format(c, i, j)
            )

    # Plugin time universal
    for (i, j) in trips:
        m.addConstr(
            p_vars[i, j] >= sigma[i, j] + d_vars[i, j] + tau[i, j],
            name='lb_plug_time_{}_{}'.format(i, j)
        )

    # Plugin time from chosen arcs
    for (c, i2, j2, i, j) in arcs_chg:
        if (i, j) == dummy_trip or (i2, j2) == dummy_trip:
            continue

        else:
            m.addConstr(
                p_vars[i, j] >= p_vars[i2, j2] + t_vars[c, i2, j2],
                name='plugin_{}_{}_{}_{}_{}'.format(c, i2, j2, i, j)
            )

    # Can't charge if x was set to 0
    for (c, i, j) in trips_skip:
        m.addConstr(
            t_vars[c, i, j] == 0,
            name='no_chg_{}_{}_{}'.format(c, i, j)
        )

    # Initial charge values
    buses = list(set(t[0] for t in trips))
    for i in buses:
        m.addConstr(
            u_vars[i, 0] == u0,
            name='init_chg_{}'.format(i))

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
            ),
            name='track_chg_{}_{}'.format(i, j)
        )

    # New solution must be better than incumbent
    eps = 0.1
    m.addConstr(
        sum(d_vars[t] for t in trips) <= best_delay - eps,
        name='beat_incumbent'
    )
    m._max_chg_time = max_chg_time
    return m


# Build the master problem
def build_master_problem(
        all_trips, chargers, delta, sigma, tau, chg_pwr,
        t_max, u_max, u0=None, delay_cutoff=1e6):
    """
    Build an instance of the Benders subproblem with Guronipy.

    :param trips: list of all trips
    :param chargers: list of available chargers
    :param delta: dict of energy consumption per trip
    :param sigma: dict of start times for all trips
    :param tau: dict of trip durations for all trips
    :param chg_pwr: dict of power output for each charger in kWh/min
    :param t_max: dict of max charge time for all trip-charger
        pairs
    :param u_max: usable battery capacity of all buses in kWh
    :param u0: initial charge of all buses in kWh (defaults to None,
        meaning fully charged)
    :param delay_cutoff: delay cutoff used by gen_arc_costs()
    :return: constructed (not solved) Gurobi LP model of SP
    """
    # Default to all buses starting out fully charged
    if u0 is None:
        u0 = u_max

    # Get arcs and costs
    dummy_trip = (-1, -1)
    # Set of trips for which charging is feasible
    feas_trip_chg_pairs = [
        (c, i, j) for c in chargers for i, j in all_trips
            if t_max[c, i, j] > 0
    ]

    # List of trips able to use each charger
    feas_trips_by_charger = {
        c: [(i, j) for cc, i, j in feas_trip_chg_pairs if cc == c]
        for c in chargers
    }
    opt_arcs, costs = gen_arc_costs(
        feas_trips=feas_trips_by_charger,
        chargers=chargers,
        start_times=sigma,
        trip_times=tau,
        delay_cutoff=delay_cutoff,
        dummy_trip=dummy_trip
    )

    m = gp.Model()
    y_vars = m.addVars(opt_arcs, obj=costs, vtype=GRB.BINARY, name='y')
    x_vars = m.addVars(feas_trip_chg_pairs, vtype=GRB.BINARY, name='x')

    # Connectivity constraints
    n_constr = 0
    for c in chargers:
        for (i, j) in feas_trips_by_charger[c]:
            in_trips = [
                (a[1], a[2]) for a in opt_arcs
                if a[0] == c and a[3:] == (i, j)]

            out_trips = [
                (a[3], a[4]) for a in opt_arcs
                if a[0] == c and a[1:3] == (i, j)]

            m.addConstr(
                sum(y_vars[c, i2, j2, i, j] for (i2, j2) in in_trips) - sum(
                    y_vars[c, i, j, i2, j2] for (i2, j2) in out_trips) == 0
            )
            n_constr += 1

    # Dummy node constraints: leave and return to depot
    for c in chargers:
        dummy_arcs_start = [
            (c, *dummy_trip, i, j) for (i, j) in feas_trips_by_charger[c]
        ]
        dummy_arcs_end = [
            (c, i, j, *dummy_trip) for (i, j) in feas_trips_by_charger[c]
        ]

        m.addConstr(
            sum(y_vars[a] for a in dummy_arcs_start) == 1)
        m.addConstr(
            sum(y_vars[a] for a in dummy_arcs_end) == 1)

    # Relationship between x and y variables
    for c in chargers:
        for (i, j) in feas_trips_by_charger[c]:
            in_trips = [
                (a[1], a[2]) for a in opt_arcs
                if a[0] == c and a[3:] == (i, j)]
            m.addConstr(
                x_vars[c, i, j] == sum(
                    y_vars[c, i2, j2, i, j] for (i2, j2) in in_trips)
            )

    # Add battery cuts
    n_cuts = 0
    vehs = list(set(i[0] for i in all_trips if i[0] != -1))
    for i in vehs:
        # Get all trips for this bus
        trips_i = sorted(j[1] for j in all_trips if j[0] == i)

        for j_init in trips_i:
            delta_sum = 0
            trips_left = sorted([j for j in trips_i if j >= j_init])
            for j in trips_left:
                delta_sum += delta[i, j]
                constr_rhs = delta_sum - u0
                if constr_rhs > 0:
                    n_cuts += 1
                    # Gather all charging opportunities that have passed
                    trip_range = [k for k in trips_i if j_init <= k < j]
                    chg_opps = [
                        (c, ii, jj) for (c, ii, jj) in feas_trip_chg_pairs
                        if ii == i and jj in trip_range
                    ]
                    # TODO: Adapt these for multiple chargers per site
                    m.addConstr(
                        sum(
                            x_vars[k] for k in chg_opps
                        ) >= np.ceil(constr_rhs / u_max),
                        'battery_{}_{}_{}'.format(i, j_init, j)
                    )

    print('Added {} cuts to master problem.'.format(n_cuts))

    m.Params.LazyConstraints = 1
    # We don't care about the bound at all for min-delay formulation,
    # we just want to get solutions quickly.
    m.Params.MIPFocus = 1
    # m.Params.LogToConsole = 0
    m._x = x_vars
    m._y = y_vars
    m._delay_incumbent = 1e12
    m._best_x = None
    m._best_y = None

    m._trips = all_trips
    m._chg_opps = feas_trip_chg_pairs
    m._delta = delta
    m._rho = chg_pwr
    m._sigma = sigma
    m._tau = tau
    m._max_chg_time = t_max
    m._u_max = u_max
    m._u0 = u0
    m._chargers = chargers

    # Counts that we track and report at the end
    m._n_benders_itr = 0
    m._n_mipsol_secs = 0
    m._n_frac_solns = 0

    return m


def extract_iis(iis_model):
    """
    Use Gurobi to extract an IIS from a model. The model should be
    infeasible.

    :param iis_model: an infeasible Gurobi LP model
    :return:
        - iis_trips, list of x-indices for constraints included in IIS
        - iis_arcs, list of y-indices for constraints included in IIS
    """
    iis_model.computeIIS()
    iis_trips = list()
    iis_arcs = list()
    for c in iis_model.getConstrs():
        if c.IISConstr:
            if 'no_chg' in c.constrname:
                # Recover index from name we gave constraint
                # Feels hacky, but will do for now.
                # TODO: this will cause errors if underscores are
                #   present in block_id strings
                c_c, v_c, i_c = c.constrname.split('_')[2:]
                # Convert strings to ints
                iis_trips.append((c_c, v_c, int(i_c)))

            if 'plugin' in c.constrname:
                # Recover index from name we gave constraint
                # Feels hacky, but will do for now.
                # TODO: this will cause errors if underscores are
                #   present in block_id strings
                c_ix = c.constrname.split('_')[1:]
                # Convert strings to ints
                iis_arcs.append(
                    (c_ix[0], c_ix[1], int(c_ix[2]), c_ix[3], int(c_ix[4]))
                )

    # Don't add indices that don't have a variable
    # (mainly a concern for trips where charging is infeasible,
    # so we never initialize an x)
    iis_trips = [
        t for t in iis_trips if iis_model._max_chg_time[t] > 0.1
    ]

    if not iis_trips and not iis_arcs:
        iis_str = ''
        for c in iis_model.getConstrs():
            if c.IISConstr:
                iis_str += c.constrname + '\n\t'

        raise ValueError(
            'Did not detect any conditional constraints in IIS. '
            'IIS contains the following constraints:\n\t{}'.format(iis_str)
        )

    return iis_trips, iis_arcs


def add_benders_constr(
        mp_model, sp_model, print_iis=False, called_from='callback'
):
    """
    Add one or more Combinatorial Benders cuts to the master problem.

    :param mp_model: master problem Gurobi model
    :param sp_model: subproblem Gurobi model
    :param print_iis: True if the calculated IIS should be printed for
        debugging
    :param called_from: string, either 'callback' or 'master'. If
        'callback', this indicates the cuts are being added within a
        Gurobi callback. If 'master', this indicates the cuts are added
        when initializing the MP (i.e., from an initial heuristic
        solution)
    :return: master problem model with Benders cuts added
    """
    max_cuts = 100
    infeas = True
    itr = 0
    while infeas and itr < max_cuts:
        itr += 1
        iis_trips, iis_arcs = extract_iis(sp_model)

        if print_iis:
            # )
            print('\tIIS {} includes {} total variables.'.format(
                itr+1, len(iis_arcs) + len(iis_trips))
            )

        # Add a constraint for the IIS that has been detected
        expr = sum(
                1 - mp_model._y[a] for a in iis_arcs
            ) + sum(
                mp_model._x[t] for t in iis_trips
            ) >= 1
        if called_from == 'callback':
            mp_model.cbLazy(expr)

        elif called_from == 'master':
            mp_model.addConstr(expr)

        else:
            raise ValueError(
                'Unrecognized called_from value: {}'.format(called_from)
            )

        # Remove one constraint from this IIS and see if model is still
        # infeasible.
        if iis_arcs:
            rmv_ix = random.choice(iis_arcs)
            rmv_name = 'plugin_{}_{}_{}_{}_{}'.format(*rmv_ix)

        else:
            rmv_ix = random.choice(iis_trips)
            rmv_name = 'no_chg_{}_{}_{}'.format(*rmv_ix)

        sp_model.remove(sp_model.getConstrByName(rmv_name))
        sp_model.optimize()
        if sp_model.status != GRB.INFEASIBLE:
            infeas = False
            if print_iis:
                print('\tModel is feasible after IIS iteration {}'.format(itr))

    return mp_model


def gen_subtour_cut(model, charger, subtour_list):
    """
    Add a subtour cut to the master problem.

    :param model: master problem Gurobi model
    :param charger: name of charger that needs a subtour cut
    :param subtour_list: list of arcs that make up subtour
    :return: MP model with subtour-eliminating cut added
    """
    # If we got more than one tour, generate a cut that corresponds to
    # the smallest subtour.
    if len(subtour_list) > 1:
        shortest_subtour = [
            st for st in subtour_list if len(st) == min(
                len(st) for st in subtour_list
            )
        ][0]
        # Add a cut that excludes this subtour
        model.cbLazy(
            sum(
                model._y[charger, *a] for a in shortest_subtour)
            <= len(shortest_subtour) - 1
        )
        return model

    else:
        print('No constraint generated for', subtour_list)


def gen_benders_cut(
        model, x_opt, y_opt, called_from='callback', debug=False):
    """
    Generate a Benders cut by solving the subproblem, updating the
    incumbent solution if necessary, and calling other functions to find
    one or more IISs and generate corresponding CB cuts.

    :param model: master problem model
    :param x_opt: dict of optimized x values from MP
    :param y_opt: dict of optimized y values from MP
    :param called_from: where this function was called from (either
        'callback' or 'master')
    :param debug: flag to print debugging information
    :return: MP model with CB cut(s) added
    """
    trips = model._trips
    chg_opps = model._chg_opps
    chargers = model._chargers
    delta = model._delta
    rho = model._rho
    sigma = model._sigma
    tau = model._tau
    max_chg_time = model._max_chg_time
    u_max = model._u_max
    u0 = model._u0

    used_arcs = [a for a in y_opt if y_opt[a] > 0.9]

    skipped_trips = [
        t for t in chg_opps if x_opt[t] < 0.1
    ]
    # Also no charging for all the trips where max charge time is 0
    # skipped_trips = skipped_trips + [
    #     (c, i, j) for c in chargers for i, j in trips
    #     if (c, i, j) not in chg_opps]

    if debug:
        # Build the subproblem
        sp_m = build_subproblem(
            trips_skip=skipped_trips,
            chargers=chargers,
            arcs_chg=used_arcs,
            trips=trips,
            delta=delta,
            chg_pwr=rho,
            sigma=sigma,
            tau=tau,
            max_chg_time=max_chg_time,
            max_chg=u_max,
            u0=u0,
            best_delay=1e100
        )
        sp_m.Params.LogToConsole = 0
        sp_m.Params.Seed = 100

        # Solve the relaxed subproblem
        sp_m.optimize()
        if sp_m.status == GRB.INFEASIBLE:
            print('SP model is infeasible without delay bound.')

        elif sp_m.status == GRB.OPTIMAL:
            # Extract objective value
            print('SP model is feasible with objective {:.2f}'.format(
                sp_m.ObjVal))

    # Build the subproblem
    sp_m = build_subproblem(
        trips_skip=skipped_trips,
        chargers=chargers,
        arcs_chg=used_arcs,
        trips=trips,
        delta=delta,
        chg_pwr=rho,
        sigma=sigma,
        tau=tau,
        max_chg_time=max_chg_time,
        max_chg=u_max,
        u0=u0,
        best_delay=model._delay_incumbent
    )
    sp_m.Params.LogToConsole = 0
    sp_m.Params.Seed = 100

    # Solve the subproblem
    sp_m.optimize()

    if sp_m.status == GRB.INFEASIBLE:
        model = add_benders_constr(
            model, sp_m, print_iis=False, called_from=called_from
        )
        return model

    elif sp_m.status == GRB.OPTIMAL:
        # Extract objective value
        sp_obj = sp_m.ObjVal

        if sp_obj > model._delay_incumbent:
            raise ValueError('Incumbent solution did not improve objective')

        # Update incumbent solution
        model._delay_incumbent = sp_obj
        model._best_x = x_opt
        model._best_y = y_opt

        print(
            'New incumbent solution found! Objective value: {:.2f}'.format(
                model._delay_incumbent
            )
        )

        # Re-solve the subproblem and generate a cut
        sp_2 = build_subproblem(
            trips_skip=skipped_trips,
            chargers=chargers,
            arcs_chg=used_arcs,
            trips=trips,
            delta=delta,
            chg_pwr=rho,
            sigma=sigma,
            tau=tau,
            max_chg_time=max_chg_time,
            max_chg=u_max,
            u0=u_max,
            best_delay=model._delay_incumbent
        )
        sp_2.Params.LogToConsole = 0

        # Solve the subproblem
        sp_2.optimize()
        model = add_benders_constr(model, sp_2, called_from=called_from)

        return model

    else:
        raise ValueError(
            'Unrecognized solver status: {}'.format(sp_m.status)
        )


def benders_callback(model, where):
    """
    Callback function used by Gurobi to run our Combinatorial Benders
    procedure within its branch-and-cut tree.

    :param model: master problem model that callback is applied to
    :param where: argument supplied by Gurobi that tells us where
        the callback is being invoked
    :return: does not return, just edits model and lets Gurobi do its
        thing
    """
    # Actions when Gurobi has found a new MIP solution
    if where == GRB.Callback.MIPSOL:
        # Extract solution
        y_opt = model.cbGetSolution(model._y)
        x_opt = model.cbGetSolution(model._x)

        used_arcs = [a for a in y_opt if y_opt[a] > 0.99]

        sec_added = False
        for c in model._chargers:
            c_arcs = [a[1:] for a in used_arcs if a[0] == c]
            # Check if there are any subtours in the solution
            subtour_list = check_for_subtours(c_arcs)
            if len(subtour_list) > 1:
                model._n_mipsol_secs += 1
                model = gen_subtour_cut(model, c, subtour_list)
                sec_added = True

        # If not, add Combinatorial Benders cut
        if not sec_added:
            model._n_benders_itr += 1
            model = gen_benders_cut(model, x_opt, y_opt)

    # Actions whenever Gurobi has solved a MIP node
    elif where == GRB.Callback.MIPNODE:
        # Check if we've found the optimal solution for this node
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)

        if status == GRB.OPTIMAL:
            # Check for an integer solution
            y_opt = model.cbGetNodeRel(model._y)
            x_opt = model.cbGetNodeRel(model._x)

            x_int = all(
                (abs(x - 0) <= 0.01)
                or (abs(x - 1) <= 0.01)
                for x in x_opt.values()
            )

            y_int = all(
                (abs(y - 0) <= 0.01)
                or (abs(y - 1) <= 0.01)
                for y in y_opt.values()
            )

            if not x_int or not y_int:
                model._n_frac_solns += 1
                # # We found an integer-feasible solution! But remember
                # # that it may have subtours
                # used_arcs = [a for a in y_opt if y_opt[a] > 0.99]
                #
                # for c in model._chargers:
                #     c_arcs = [a[1:] for a in used_arcs if a[0] == c]
                #     # Check if there are any subtours in the solution
                #     subtour_list = check_for_subtours(c_arcs)
                #     if len(subtour_list) > 1:
                #         model._n_mipsol_secs += 1
                #         model = gen_subtour_cut(model, c, subtour_list)


def check_for_subtours(opt_edges):
    """
    Check whether the given set of arcs includes a subtour. If so,
    return all such subtours.

    :param opt_edges: list of arcs included in optimal solution as
        5-tuples of charger and trips
    :return: list of any subtours in the solution (each as a list of
        5-tuple arcs)
    """
    all_tours = list()
    subtour = list()
    edges_left = opt_edges.copy()
    while edges_left:
        start_edge = edges_left[0]
        next_node = start_edge[2:]
        subtour.append(start_edge)
        edges_left.remove(start_edge)
        while next_node != start_edge[:2]:
            next_edge = [a for a in edges_left if a[:2] == next_node][0]
            subtour.append(next_edge)
            edges_left.remove(next_edge)
            next_node = next_edge[:2] if next_edge[2:] == next_node \
                else next_edge[2:]
        all_tours.append(subtour)
        subtour = list()
    return all_tours


def solve_with_benders(
        case_data, chargers, rho, u_max, heur_solns=None, cut_gap=0.5,
        time_limit=7200, delay_cutoff=60):
    """
    Solve the provided charge scheduling instance with Combinatorial
    Benders.

    :param case_data: dict of problem parameters
    :param chargers: list of chargers available
    :param rho: dict giving power output of all chargers (in kWh/min)
    :param u_max: usable capacity of batteries in kWh
    :param heur_solns: dict of heuristic solutions if any have been
        generated, used for initial incumbent and CB cuts
    :param cut_gap: relative optimality gap to use a heuristic solution
        for generating a cut. If the best heuristic solution has
        objective value 100 and cut_gap=0.5, any solution with delay
        <= 150 will be included
    :param time_limit: solver time limit
    :param delay_cutoff: delay cutoff used in network generation with
        gen_arc_costs()
    :return: prints optimization results
    """
    # Create master problem
    mp_m = build_master_problem(
        all_trips=case_data['trips'],
        chargers=chargers,
        sigma=case_data['sigma'],
        tau=case_data['tau'],
        chg_pwr=rho,
        t_max=case_data['max_chg_time'],
        u_max=u_max,
        delta=case_data['delta'],
        delay_cutoff=delay_cutoff
    )

    if heur_solns:
        sorted_solns = sorted(heur_solns, key=heur_solns.get)
        best_obj = heur_solns[sorted_solns[0]]
        mp_m._delay_incumbent = best_obj
        n_good_solns = 0
        for itr, arc_list in enumerate(sorted_solns):
            itr_obj = heur_solns[arc_list]
            if (itr_obj - best_obj) / best_obj <= cut_gap:
                logger.debug(
                    'Adding cut for heuristic solution with objective value'
                    ' {:.2f}'.format(itr_obj))
                n_good_solns += 1
                # Convert arcs and trips to x and y values. First, add
                # dummy arcs for each charger.
                all_used_arcs = list()
                for c in chargers:
                    c_arcs = [a for a in arc_list if a[0] == c]
                    c_out = (c, -1, -1, c_arcs[0][1], c_arcs[0][2])
                    c_in = (c, c_arcs[-1][3], c_arcs[-1][4], -1, -1)
                    all_used_arcs += [c_out] + c_arcs + [c_in]

                # Extract all trips selected for charging
                chg_opps_used = [
                    a[:3] for a in all_used_arcs[1:]
                ]

                # Create mock x and y variables
                x_itr = {t: 1 if t in chg_opps_used else 0 for t in mp_m._x}
                y_itr = {a: 1 if a in all_used_arcs else 0 for a in mp_m._y}
                # Generate a CB cut
                mp_m = gen_benders_cut(mp_m, x_itr, y_itr, called_from='master')

        print(
            'Added Combinatorial Benders cuts for {} sufficiently good '
            'heuristic solutions.'.format(n_good_solns)
        )

    mp_m.Params.TimeLimit = time_limit
    mp_m.optimize(benders_callback)
    print('Optimal delay: {:.2f}'.format(mp_m._delay_incumbent))
    print('Number of SECs: {}'.format(
        mp_m._n_mipsol_secs
    ))
    print('Number of Benders iterations: {}'.format(
        mp_m._n_benders_itr
    ))
    print('Number of fractional solutions: {}'.format(mp_m._n_frac_solns))


def to_df(model, sigma_dict, tau_dict):
    """
    Save results of CB model as a Pandas DataFrame.

    :param model: complete or subproblem model (only continuous
        variables are needed)
    :param sigma_dict: dict of trip start times
    :param tau_dict: dict of trip durations
    :return: pandas DataFrame of results
    """
    t_opt = model.getAttr('X', model._t)
    d_opt = model.getAttr('X', model._d)
    u_opt = model.getAttr('X', model._u)
    p_opt = model.getAttr('X', model._p)

    v_list = list()
    t_list = list()
    c_list = list()
    start_times = list()
    end_times = list()
    opt_chg_times = list()
    opt_soc = list()
    opt_plugin = list()
    opt_delay = list()
    for (v, t) in model._trips:
        for c in model._chargers:
            v_list.append(v)
            t_list.append(t)
            c_list.append(c)
            start_times.append(sigma_dict[v, t])
            end_times.append(sigma_dict[v, t] + tau_dict[v, t])
            try:
                opt_chg_times.append(t_opt[c, v, t])
            except KeyError:
                opt_chg_times.append(0)
            try:
                opt_soc.append(u_opt[v, t])
            except ValueError:
                opt_soc.append(np.nan)
            opt_plugin.append(p_opt[v, t])
            opt_delay.append(d_opt[v, t])

    out_dict = {
        'block_id': v_list,
        'trip_idx': t_list,
        'charger': c_list,
        'soc': opt_soc,
        'start_time': start_times,
        'end_time': end_times,
        'plugin_time': opt_plugin,
        'chg_time': opt_chg_times,
        'delay': opt_delay
    }
    out_df = pd.DataFrame(out_dict)
    out_df = out_df.sort_values(by=['block_id', 'trip_idx', 'charger'])
    return out_df




