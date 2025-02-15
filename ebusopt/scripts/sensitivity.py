from ebusopt.opt.charger_location import ChargerLocationModel
from numpy import linspace
import pickle
import logging
import copy

"""
Sensitivity analysis from TR Part C paper.
"""


def run_alpha_sensitivity(flm, min_alpha, max_alpha, num_runs, save_prefix):
    flm.check_charging_needs()
    flm.set_conflict_sets()
    for i, a in enumerate(linspace(min_alpha, max_alpha, num_runs)):
        logging.info('Model solve {} of {}'.format(i+1, num_runs))
        logging.info('Parameter value: {}'.format(a))
        flm.build_and_solve(alpha=a)
        save_fname = '../results/sensitivity/{}_{}.csv'.format(
            save_prefix, i)
        flm.summary_to_csv(save_fname, 'alpha', a)


def run_power_sensitivity(flm, min_power, max_power, num_runs):
    # flm.max_chargers = {s: 200 for s in flm.chg_sites}

    for i, p in enumerate(linspace(min_power, max_power, num_runs)):
        flm_i = copy.deepcopy(flm)
        flm_i.chg_rates = {s: p / 60 for s in flm_i.chg_sites}
        logging.info('Model solve {} of {}'.format(i+1, num_runs))
        logging.info('Parameter value: {}'.format(p))
        flm_i.solve(alpha=2000, simple_case=True, bu_kwh=400)
        # flm_i.log_results()
        save_fname = '../results/sensitivity/rho_{}.csv'.format(i)
        flm_i.summary_to_csv(save_fname, 'rho', p)


def run_battery_sensitivity(flm, min_u, max_u, num_runs):
    # flm.chg_rates = {s: 100 / 60 for s in flm.chg_sites}
    # flm.max_chargers = {s: 20 for s in flm.chg_sites}
    for i, u in enumerate(linspace(min_u, max_u, num_runs)):
        flm_i = copy.deepcopy(flm)
        flm_i.chg_lims = {v: u for v in flm_i.vehicles}
        logging.info('Model solve {} of {}'.format(i+1, num_runs))
        logging.info('Parameter value: {}'.format(u))
        flm_i.solve(alpha=2000, simple_case=True, bu_kwh=400)
        # flm_i.log_results()
        save_fname = '../results/sensitivity/u_max_{}.csv'.format(i)
        flm_i.summary_to_csv(save_fname, 'u_max', u)


def run_power_and_battery_sensitivity(
        flm, min_u, max_u, min_power, max_power, num_runs):
    # flm.max_chargers = {s: 20 for s in flm.chg_sites}
    # Sensitivity for both power and battery combined
    for j, u in enumerate(linspace(min_u, max_u, num_runs)):
        for i, p in enumerate(linspace(min_power, max_power, num_runs)):
            flm_i = copy.deepcopy(flm)
            flm_i.chg_rates = {s: p / 60 for s in flm_i.chg_sites}
            flm_i.chg_lims = {v: u for v in flm_i.vehicles}
            iter_no = j*num_runs + i + 1
            logging.info('Model solve {} of {}'.format(iter_no, num_runs**2))
            logging.info('Battery capacity: {:.2f}'.format(u))
            logging.info('Power value: {:.2f}'.format(p))
            flm_i.solve(alpha=2000, simple_case=True, bu_kwh=400)
            # flm_i.log_results()
            save_fname = '../results/sensitivity/combined_rho_u_{}.csv'.format(
                iter_no)
            flm_i.summary_to_csv(save_fname, 'rho', p, 'u_max', u)


def run_g_sensitivity(flm, min_g, max_g, num_runs):
    flm.check_charging_needs()
    flm.set_conflict_sets()
    for i, g in enumerate(linspace(min_g, max_g, num_runs)):
        flm.charger_costs = {s: g for s in flm.chg_sites}
        logging.info('Model solve {} of {}'.format(i+1, num_runs))
        logging.info('Parameter value: {}'.format(g))
        flm.build_and_solve(alpha=1)
        # flm.log_results()
        save_fname = '../results/sensitivity/g_{}.csv'.format(i)
        flm.summary_to_csv(save_fname, 'g', g)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with open('simple_case.pickle', 'rb') as f:
        opt_kwargs = pickle.load(f)
    flm = ChargerLocationModel(**opt_kwargs)

    # run_battery_sensitivity(flm, 100, 500, 50)
    # run_power_sensitivity(flm, 50, 600, 50)
    # run_alpha_sensitivity(flm, 100, 8000, 50, 'alpha')
    run_power_and_battery_sensitivity(
        flm, 100, 500, 50, 600, 10)
    # run_g_sensitivity(flm, 1, 5000, 50)

