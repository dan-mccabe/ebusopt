import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import os


def plot_stacked_costs(df, x, x_label, alpha, gamma, legend_loc='center'):
    time_cost = alpha * df['recov_time_lost']
    bu_cost = gamma * df['n_backups']
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.stackplot(df[x], df['capital_cost'], time_cost, bu_cost,
                 labels=('Capital cost', 'Charging cost', 'Backup bus cost'),
                 colors=('r', 'b', 'g'))
    ax.set_xlabel(x_label)
    ax.set_ylabel('Cost')
    ax.legend(loc=legend_loc)
    fig.tight_layout()

    return fig


def plot_one_column(df, x, x_label, y, y_label, marker='o', int_ticks=True):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df[x], df[y], label=y_label, color='r', linestyle='solid', marker=marker)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def plot_two_columns(df, x, x_label, y1, y1_label, y2, y2_label,
                     save_name, legend_loc='center', yscale='diff'):
    if yscale == 'diff':
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y1_label)
        ax1.plot(df[x], df[y1], label=y1_label, color='r', linestyle='solid')

        ax2 = ax1.twinx()
        ax2.set_ylabel(y2_label)
        ax2.plot(df[x], df[y2], label=y2_label, color='b', linestyle='dashed')
        fig.legend(loc=legend_loc)
        fig.tight_layout()
        fig.savefig(save_name, dpi=600)
        return fig

    elif yscale == 'same':
        plt.plot(df[x], df[y1], label=y1_label, color='r', linestyle='solid')
        plt.plot(df[x], df[y2], label=y2_label, color='b', linestyle='dashed')
        plt.ylabel('Time (min)')
        plt.xlabel(x_label)
        plt.legend(loc=legend_loc)
        plt.tight_layout()
        plt.savefig(save_name, dpi=600)


def plot_num_chargers(df, xcol, xlabel):
    n_cols = [c for c in df.columns if
              c[:4] == 'N at' and (df[c] > 1e-3).any()]
    n_labs = {c: c[4:] for c in n_cols}

    fig, ax = plt.subplots(figsize=(9, 6))
    patterns = ['solid', 'dotted', 'dashed', 'dashdot',
                (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5))]
    for i, c in enumerate(n_cols):
        ax.plot(df[xcol], df[c], label=n_labs[c], linestyle=patterns[i])

    plt.ylabel('Number of Chargers Built')
    plt.xlabel(xlabel)
    plt.legend(loc='best')
    plt.tight_layout()
    return fig


def plot_costs_and_num_chargers(df, x, xlabel, alpha=None, gamma=200,
                                legend_loc_a='best', legend_loc_b='best'):
    # First plot: stacked costs
    fig = plt.figure(figsize=(16, 9))
    ax1 = plt.subplot(1, 2, 1)
    time_cost = df['operations_cost']
    #     if x == 'alpha':
    #         time_cost = df['alpha'] * df['recov_time_lost']
    #     else:
    #         time_cost = alpha * df['recov_time_lost']

    bu_cost = gamma * df['n_backups']
    if any(bu_cost > 1e-6):
        y_plot = (df['capital_cost'], time_cost, bu_cost)
        labels = ('Capital cost', 'Charging cost', 'Backup bus cost')
    else:
        y_plot = (df['capital_cost'], time_cost)
        labels = ('Capital cost', 'Charging cost')

    ax1.stackplot(df[x], *y_plot,
                  labels=labels,
                  colors=('r', 'b', 'g'))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Cost')
    ax1.legend(loc=legend_loc_a)
    ax1.set_title('(a)')

    n_cols = [c for c in df.columns if
              c[:4] == 'N at' and (df[c] > 1e-3).any()]
    n_labs = {c: c[4:] for c in n_cols}

    # Second plot: number of chargers
    ax2 = plt.subplot(1, 2, 2)
    patterns = ['solid', 'dotted', 'dashed', 'dashdot',
                (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5))]
    for i, c in enumerate(n_cols):
        ax2.plot(df[x], df[c], label=n_labs[c], linestyle=patterns[i])

    ax2.set_ylabel('Number of Chargers Built')
    ax2.set_xlabel(xlabel)
    ax2.legend(loc=legend_loc_b)
    ax2.set_title('(b)')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    return fig


def plot_three_sensitivity(df, x, xlabel, alpha, gamma):
    # First plot: stacked costs
    fig = plt.figure(figsize=(16, 9))
    ax1 = plt.subplot(1, 3, 2)
    time_cost = df['operations_cost']
    #     time_cost = alpha * df['recov_time_lost']
    bu_cost = gamma * df['n_backups']
    ax1.stackplot(df[x], df['capital_cost'], time_cost, bu_cost,
                  labels=('Capital cost', 'Charging cost', 'Backup bus cost'),
                  colors=('r', 'b', 'g'))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Cost')
    ax1.legend(loc='best')
    ax1.set_title('(b)')

    # Second plot: number of chargers built
    n_cols = [c for c in df.columns if
              c[:4] == 'N at' and (df[c] > 1e-3).any()]
    n_labs = {c: c[4:] for c in n_cols}
    ax2 = plt.subplot(1, 3, 3)
    patterns = ['solid', 'dotted', 'dashed', 'dashdot',
                (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5))]
    for i, c in enumerate(n_cols):
        ax2.plot(df[x], df[c], label=n_labs[c], linestyle=patterns[i])

    ax2.set_ylabel('Number of Chargers Built')
    ax2.set_xlabel(xlabel)
    ax2.legend(loc='best')
    ax2.set_title('(c)')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Third plot: number of backup buses used
    ax3 = plt.subplot(1, 3, 1)
    ax3.plot(df[x], df['n_backups'], color='r', linestyle='solid')
    ax3.set_ylabel('Number of Backup Buses Used')
    ax3.set_xlabel(xlabel)
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.set_title('(a)')
    plt.tight_layout()

    return fig


def plot_u_rho_sensitivity():
    all_files = os.listdir(prefix)
    combined_files = [prefix + '/' + f for f in all_files if
                      f[:8] == 'combined' and f[-3:] == 'csv']
    # Read in results
    comb_dfs = list()
    for f in combined_files:
        comb_dfs.append(pd.read_csv(f))
    comb_df = pd.concat(comb_dfs).sort_values(by=['u_max', 'rho'])

    rho_vals = comb_df['rho'].unique()
    u_vals = comb_df['u_max'].unique()
    comb_df = comb_df.set_index(['u_max', 'rho'])
    comb_df['total_cost'] = comb_df['obj_val'] + 200*comb_df['n_backups']

    n = 10
    cvals = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            val = comb_df.loc[(u_vals[i], rho_vals[j]), 'total_cost']
            if isinstance(val, np.float64) or isinstance(val, float):
                cvals[i, j] = val
            else:
                print(u_vals[i], rho_vals[j], val)
                cvals[i, j] = val[0]

    fig = plt.figure(figsize=(9, 6))
    plt.pcolor(u_vals[:n], rho_vals[:n], np.log(cvals))
    plt.xlabel('$\overline{u}_v$ (kWh)')
    plt.ylabel(r'$\rho^s$ (kW)')
    plt.colorbar(label='Logarithm of cost')
    plt.tight_layout()
    plt.savefig('../results/power_and_capacity_sensitivity.pdf', dpi=350)
    return fig


def plot_alpha_sensitivity():
    all_files = os.listdir(prefix)
    alpha_files = [prefix + '/' + f for f in all_files if f[:5] == 'alpha' and f[-3:] == 'csv']
    # Read in results for varying alpha
    alpha_dfs = list()
    for f in alpha_files:
        alpha_dfs.append(pd.read_csv(f))
    alpha_df = pd.concat(alpha_dfs).sort_values(by='alpha')
    alpha_fig_dbl = plot_costs_and_num_chargers(
        alpha_df, 'alpha', r'$\alpha$', legend_loc_a='upper left')
    alpha_fig_dbl.savefig('../results/alpha_sensitivity_full.pdf', dpi=350)


if __name__ == '__main__':
    # Plot settings
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "Times",
        "font.serif": "Times",
        "font.sans-serif": ["Times"],
        'font.size': 16})

    prefix = '../results/sensitivity'

    # Charger power results
    all_files = os.listdir(prefix)
    rho_files = [prefix + '/' + f for f in all_files if
                 f[:3] == 'rho' and f[-3:] == 'csv']
    rho_dfs = list()
    for f in rho_files:
        rho_dfs.append(pd.read_csv(f))
    rho_df = pd.concat(rho_dfs).sort_values(by='rho')
    # rho_fig = plot_stacked_costs(rho_df, 'rho', r'$\rho$ (kW)', 2, 200)
    # rho_fig.savefig('../results/rho_costs.pdf', dpi=350)

    rho_fig_triple = plot_three_sensitivity(
        rho_df, 'rho', r'$\rho^s$ (kW)', 2, 200)
    rho_fig_triple.savefig('../results/rho_sensitivity_full.pdf', dpi=350)

    # Battery capacity
    u_files = [prefix + '/' + f for f in all_files if
               f[:5] == 'u_max' and f[-3:] == 'csv']
    u_dfs = list()
    for f in u_files:
        u_dfs.append(pd.read_csv(f))
    u_df = pd.concat(u_dfs).sort_values(by='u_max')

    # u_costs = plot_stacked_costs(
    #     u_df, 'u_max', r'$\overline{u}$ (kW)', 2, 200, 'best')
    # u_costs.savefig('../results/u_costs.pdf', dpi=350)
    u_full_fig = plot_three_sensitivity(
        u_df, 'u_max', r'$\overline{u}_v$ (kWh)', 2, 200)
    u_full_fig.savefig('../results/u_sensitivity_full.pdf', dpi=350)

    plot_u_rho_sensitivity()
    plot_alpha_sensitivity()
