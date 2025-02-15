import numpy as np
import pandas as pd
import datetime
import plotly.express as px
import time


class LoadProfile:
    def __init__(self, srs: pd.Series, name: str):
        self.load_srs = srs
        self.name = name

    def __add__(self, other):
        """Add two LoadProfile objects together"""
        new_load_df = self.load_srs.add(other.load_srs.copy(), fill_value=0)
        new_name = self.name + ' + ' + other.name
        return LoadProfile(new_load_df, new_name)

    @staticmethod
    def from_list(lp_list, name):
        """
        Create a composite LoadProfile from a list of LoadProfile objects.
        """
        full_df = lp_list[0].load_srs.copy()
        for df in lp_list[1:]:
            full_df += df.load_srs
        return LoadProfile(full_df, name)

    def max_kw(self):
        return self.load_srs.max()

    def total_kwh(self):
        return self.load_srs.sum() / 60

    def load_factor(self):
        total_hrs = (
            self.load_srs.index.max() - self.load_srs.index.min()
        ).total_seconds() / 3600
        avg_kw = self.total_kwh() / total_hrs
        return avg_kw / self.max_kw()

    def to_csv(self, path_or_buf=None):
        return self.load_srs.to_csv(
            path_or_buf=path_or_buf, index_label='time'
        )

    def filter_datetime(self, min_dt, max_dt, inner=True, inplace=False):
        """Filter out load outside the min and max datetimes provided"""
        inner_idx = (self.load_srs.index >= min_dt) \
                & (self.load_srs.index <= max_dt)
        if inner:
            new_df = self.load_srs[inner_idx].copy()
        else:
            new_df = self.load_srs[~inner_idx].copy()

        if inplace:
            self.load_srs = new_df
        else:
            return LoadProfile(new_df, '')

    def rename(self, new_name):
        self.name = new_name

    def plot(self, freq=None):
        if freq is None:
            sampled_power = self.load_srs
        else:
            sampled_power = self.load_srs.resample(freq).mean()

        return px.line(
            sampled_power,
            y='power',
            labels={'index': 'Time', 'power': 'Average Load (kW)'},
            title='Load Profile: {}'.format(self.name),
        ).update_layout(xaxis_title=None)

    def calculate_costs(
            self, peak_start=6, peak_end=22, peak_price=0.1029,
            off_peak_price=0.0577,
            peak_demand_charge=4.88, off_peak_demand_charge=0.31
    ):
        """Not currently used, but not removing yet"""
        off_peak_flag = (self.load_srs.index.hour < peak_start) | (
                self.load_srs.index.hour > peak_end)
        off_peak_df = self.load_srs[off_peak_flag]
        peak_df = self.load_srs[~off_peak_flag]

        # Peak cost
        peak_energy_cost = peak_price * peak_df.sum() / 60
        peak_demand_cost = peak_demand_charge * peak_df.max()
        # Off-peak costs
        off_peak_energy_cost = off_peak_price * off_peak_df.sum() / 60
        off_peak_demand_cost = off_peak_demand_charge * off_peak_df.max()

        return pd.DataFrame(
            {
                'Normalized Total Daily Cost': [
                    peak_energy_cost + off_peak_energy_cost + (
                            (peak_demand_cost + off_peak_demand_cost)
                            * 12 / 365.25)],
                'Total TOU Energy Cost': [
                    peak_energy_cost + off_peak_energy_cost],
                'Total Demand Charge': [
                    peak_demand_cost + off_peak_demand_cost],
                'Peak Energy Cost': [peak_energy_cost],
                'Peak Demand Cost': [peak_demand_cost],
                'Off-Peak Energy Cost': [off_peak_energy_cost],
                'Off-Peak Demand Cost': [off_peak_demand_cost]
            }
        )

    def summarize(self):
        # Load factor
        total_kwh = self.load_srs.sum() / 60
        total_hrs = (
                            self.load_srs.index.max() - self.load_srs.index.min()
        ).total_seconds() / 3600
        avg_kw = total_kwh / total_hrs
        max_kw = self.load_srs.max()
        load_factor = avg_kw / max_kw
        return pd.DataFrame(
            {
                'Total kWh': [total_kwh],
                'Average kW': [avg_kw],
                'Peak kW': [max_kw],
                'Load Factor': [load_factor]
            }
        )


def gen_charger_load(
        result_df, charger_name, min_time=None, max_time=None
):
    charger_df = result_df[result_df['charger'] == charger_name]
    # Round everything to the minute. Be conservative to avoid overlap.
    charger_df['plugin_time'] = charger_df['plugin_time'].dt.round(
        '1min', 'up')
    charger_df['finish_chg_time'] = charger_df['finish_chg_time'].dt.round(
        '1min', 'down')

    # New idea: create time index first. Iterate through rows, adding
    # to load as we go.
    if min_time is None:
        min_time = result_df['plugin_time'].min().round('1min', 'down')
    if max_time is None:
        max_time = result_df['finish_chg_time'].max().round('1min', 'up')
    load_srs = pd.Series(
        index=pd.date_range(
            start=min_time,
            end=max_time,
            freq='1min'
        ),
        data=0.,
        name='power'
    )

    for _, rw in charger_df.iterrows():
        # TODO: check edge cases
        add_idx = pd.date_range(
            start=rw['plugin_time'], end=rw['finish_chg_time'], freq='1min'
        )
        load_srs.loc[add_idx] += rw['power']
    return load_srs


def set_depot_charging_plan(
        depot_df, method, peak_hour_start=None, peak_hour_end=None):
    if method == 'min_power':
        # Method 1: charge at min power for full available duration
        depot_df['depot_time'] = depot_df['first_trip_start'] - depot_df[
            'last_trip_end']
        depot_df['depot_hrs'] = depot_df['depot_time'].dt.total_seconds() / 3600
        depot_df['power'] = depot_df['kwh_to_gain'] / depot_df['depot_hrs']
        depot_df = depot_df.rename(
            columns={
                'last_trip_end': 'plugin_time',
                'first_trip_start': 'finish_chg_time'
            }
        )

    elif method == '60kW':
        # Method 2: charge at 60 kW until done, or higher as necessary
        depot_df['power'] = 60.
        depot_df['finish_chg_time'] = \
            depot_df['last_trip_end'] + pd.to_timedelta(
                depot_df['kwh_to_gain'] / 60, unit='h'
            )
        depot_df['depot_hrs'] = (depot_df['first_trip_start'] - depot_df[
            'last_trip_end']).dt.total_seconds() / 3600
        more_power_idx = depot_df['finish_chg_time'] > depot_df[
            'first_trip_start']
        new_power = depot_df.loc[more_power_idx, 'kwh_to_gain'] / depot_df.loc[
            more_power_idx, 'depot_hrs']
        depot_df.loc[more_power_idx, 'power'] = new_power
        depot_df.loc[more_power_idx, 'finish_chg_time'] = depot_df.loc[
            more_power_idx, 'first_trip_start']
        depot_df = depot_df.rename(
            columns={'last_trip_end': 'plugin_time'}
        )

    elif method == 'off_peak':
        # First step: how much off-peak time is available?
        # TODO: clean up datetime handling throughout
        min_date = depot_df['first_trip_start'].min().date()
        if peak_hour_end is None:
            # Default to 10pm
            off_peak_start = datetime.datetime(
                min_date.year, min_date.month, min_date.day, 22) \
                - datetime.timedelta(days=1)
        else:
            off_peak_start = peak_hour_end
        depot_df['off_peak_start'] = off_peak_start

        depot_df['plugin_time'] = depot_df[
            ['off_peak_start', 'last_trip_end']
        ].max(axis=1)

        if peak_hour_start is None:
            # Default to 8 hours after start
            off_peak_end = off_peak_start + datetime.timedelta(hours=8)
        else:
            off_peak_end = peak_hour_start + datetime.timedelta(days=1)
        # TODO: handle cases where a block ends after 6 am, etc. No edge
        #   cases yet so being lazy for now
        depot_df['off_peak_end'] = off_peak_end
        depot_df['finish_chg_time'] = depot_df[
            ['off_peak_end', 'first_trip_start']].min(axis=1) - datetime.timedelta(minutes=1)
        depot_df['off_peak_hrs'] = (depot_df['finish_chg_time'] - depot_df[
            'plugin_time']).dt.total_seconds() / 3600
        depot_df['power'] = depot_df['kwh_to_gain'] / depot_df['off_peak_hrs']

    else:
        method_vals = ['min_power', 'off_peak', '60kW']
        raise ValueError(
            '{} is not a recognized method. Valid options: {}'.format(
                method, method_vals
            )
        )

    depot_df['charger'] = 'Depot'
    return depot_df[['plugin_time', 'finish_chg_time', 'power', 'charger']]


def create_depot_load(
        df, zero_time, method, kwh_per_mi, max_kwh, peak_hour_start=None,
        peak_hour_end=None, min_time=None, max_time=None):
    """
    Estimate depot charging load for all blocks in the given dataframe
    of trips.
    """
    df = df.copy()[
        ['block_id', 'trip_idx', 'start_time', 'end_time', 'total_dist',
         'dh_dist']
    ]
    df['block_id'] = df['block_id'].astype(str)
    df['kwh'] = kwh_per_mi * (df['total_dist'] + df['dh_dist'])
    for time_col in ['start_time', 'end_time']:
        if not np.issubdtype(df[time_col].dtype, np.datetime64):
            df[time_col] = zero_time + pd.to_timedelta(
                df[time_col], unit='minute')

    finish_kwh = df.groupby('block_id')['kwh'].sum() - max_kwh
    finish_kwh[finish_kwh < 0] = 0

    # Extract last trip info
    last_trips = df.groupby('block_id')['trip_idx'].max().reset_index()
    last_trips_ix = pd.MultiIndex.from_frame(last_trips)
    df_last = df.set_index(['block_id', 'trip_idx']).loc[last_trips_ix]
    df_last = df_last.rename(
        columns={'end_time': 'last_trip_end'}
    )

    # Clean up
    df_last = df_last.reset_index()[
        ['block_id', 'last_trip_end']
    ].set_index('block_id')
    df_last.loc[:, 'final_soc'] = finish_kwh
    # TODO: be careful about indexing from 0 vs 1 here and in opt. code
    # df_last['trip_idx'] += 1

    # Get first trip info and clean up
    df_first = df[df['trip_idx'] == 1][
        ['block_id', 'start_time']
    ].rename(
        columns={'start_time': 'first_trip_start'}
    ).set_index('block_id')
    df_first['initial_soc'] = max_kwh
    # Merge together info by block
    depot_df = df_first.join(df_last)
    depot_df['kwh_to_gain'] = depot_df['initial_soc'] - depot_df['final_soc']
    depot_df['first_trip_start'] += datetime.timedelta(days=1)

    depot_merge = set_depot_charging_plan(
        depot_df, method=method,
        peak_hour_start=peak_hour_start, peak_hour_end=peak_hour_end
    )
    load_df = gen_charger_load(
        depot_merge, 'Depot', min_time=min_time, max_time=max_time
    )
    return LoadProfile(load_df, 'Depot')


def create_opportunity_load(
        df, kwh_per_mi, max_kwh, min_time=None, max_time=None):
    """
    Estimate opportunity charging load for all blocks in the given
    dataframe of trips.
    """
    df = df.copy()[
        ['block_id', 'trip_idx', 'start_time', 'end_time', 'total_dist',
         'dh_dist']
    ]
    df['block_id'] = df['block_id'].astype(str)
    df['kwh'] = kwh_per_mi * (df['total_dist'] + df['dh_dist'])

    # for time_col in ['start_time', 'end_time']:
    #     if not np.issubdtype(df[time_col].dtype, np.datetime64):
    #         df[time_col] = zero_time + pd.to_timedelta(
    #             df[time_col], unit='minute')

    # Filter down to just blocks that need opportunity charging
    # TODO: make sure it's consistent to number reported elsewhere
    oppo_kwh = df.groupby('block_id')['kwh'].sum()
    oppo_blocks = oppo_kwh[oppo_kwh > max_kwh].index.tolist()
    df = df[df['block_id'].isin(oppo_blocks)]

    # Compile layover time info
    # This is gonna require grouping by block and some shift() calls...
    # sounds like a task for tomorrow!
    block_gb = df.groupby('block_id')
    df_list = list()
    for _, block_df in block_gb:
        block_df = block_df.sort_values(by='trip_idx')
        block_df['next_trip_start'] = block_df['start_time'].shift(-1)
        block_df['layover_time'] = block_df['next_trip_start'] \
            - block_df['end_time']
        block_df['layover_time'] = block_df[
                                       'layover_time'].dt.total_seconds() / 60
        block_df['layover_time'] = block_df['layover_time'].fillna(0)
        df_list.append(block_df)

    df = pd.concat(df_list)
    # Use total layover per block to determine charging time/power
    block_summaries = df.groupby('block_id')[
        ['layover_time', 'kwh']].sum()
    # Divide kWh by hours to get power needed
    block_summaries['power'] = block_summaries['kwh'] / (
            block_summaries['layover_time'] / 60)
    # Merge into the rest of the DF
    df = df.merge(block_summaries[['power']], left_on='block_id', right_index=True)
    # Remove non-charging trips (generally, the last trip of the day)
    df = df[df['layover_time'] > 0]
    df['charger'] = 'Opportunity'
    df = df.rename(
        columns={
            'end_time': 'plugin_time',
            'next_trip_start': 'finish_chg_time'
        }
    )

    load_df = gen_charger_load(
        df, 'Opportunity', min_time=min_time, max_time=max_time
    )
    return LoadProfile(load_df, 'Opportunity')



