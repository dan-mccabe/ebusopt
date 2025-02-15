import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from millify import millify
from ebusopt.zebra.utils import config_page, read_markdown
from ebusopt.zebra.load import create_depot_load, LoadProfile, create_opportunity_load

st.set_page_config(
    page_title='Energy Costs',
    page_icon='🦓',
    layout='wide'
)
config_page()
st.markdown(
    "<h2 style='text-align: center; color: black;'>Energy Costs</h2>",
    unsafe_allow_html=True
)


@st.cache_data(show_spinner=False, ttl=900)
def get_depot_load_cached(
        date, routes, method, kwh_per_mi, battery_kwh, min_time, max_time,
        peak_start, peak_end
):
    # Note: routes is included only for caching purposes
    lp = create_depot_load(
        st.session_state.beb_trips, date,
        method=method, kwh_per_mi=kwh_per_mi,
        max_kwh=battery_kwh,
        min_time=min_time, max_time=max_time,
        peak_hour_start=peak_start,
        peak_hour_end=peak_end
    )
    return lp


@st.cache_data(show_spinner=False, ttl=900)
def get_oppo_load_cached(
        date, routes, kwh_per_mi, max_kwh, min_time, max_time
):
    return create_opportunity_load(
        df=st.session_state.beb_trips, kwh_per_mi=kwh_per_mi,
        max_kwh=max_kwh, min_time=min_time, max_time=max_time
    )


def reset_rate_structure():
    if st.session_state.energy_rate_type == 'Constant':
        if st.session_state.charge_method == 'Prioritize Off-Peak Charging':
            st.session_state.charge_method = 'Charge for All Available Time'

        st.session_state.tou_demand_charge = False

if 'gtfs_run' not in st.session_state:
    st.error('Please specify GTFS data before visiting this screen.')
    st.stop()

if not st.session_state.gtfs_run:
    st.error('Please specify GTFS data before visiting this screen.')
    st.stop()

if not st.session_state.range_set:
    st.error(
        'Please set bus range on the "Range Requirements" page before visiting'
        ' this screen.')
    st.stop()

dt_set = st.session_state.date_set.to_pydatetime()
default_start = dt_set + datetime.timedelta(hours=6)
default_end = dt_set + datetime.timedelta(hours=22)
price_defaults = {
    'peak_kwh_price': 0.1108,
    'off_peak_kwh_price': 0.0622,
    'demand_charge': 5.,
    'constant_kwh_price': 0.10,
    'energy_rate_type': 'Time of Use (TOU)',
    'rate_includes_demand_charge': True,
    'tou_demand_charge': True,
    'charge_method': 'Charge for All Available Time',
    'peak_hrs': (default_start, default_end),
    'kwh_per_mi': 3.,
    'battery_kwh': 450.,
    'peak_demand_charge': 5.26,
    'off_peak_demand_charge': 0.33
}
for k in price_defaults:
    if k not in st.session_state:
        st.session_state[k] = price_defaults[k]


home_tab, cost_tab, load_tab, price_tab, help_tab = st.tabs(
    ['Home', '💵 Cost Estimates', '📈 Load Profile', '⚙️ Adjust Prices', '❓ Help']
)

with price_tab:
    st.markdown('## Adjust Prices')

    with st.columns(3)[0]:
        st.selectbox(
            'Rate Structure',
            options=['Constant', 'Time of Use (TOU)'],
            key='energy_rate_type',
            on_change=reset_rate_structure
        )

    if st.session_state.energy_rate_type == 'Time of Use (TOU)':
        with st.container(border=True):
            min_time = dt_set + datetime.timedelta(hours=0)
            max_time = dt_set + datetime.timedelta(hours=24)
            st.markdown('#### Set Peak Period Hours')
            st.slider(
                label='Peak Period Hours',
                label_visibility='hidden',
                min_value=min_time,
                max_value=max_time,
                step=datetime.timedelta(hours=1),
                format='LT',
                key='peak_hrs'
            )

            st.write(
                'All energy consumption between {:%-I:%M %p} and {:%-I:%M %p} '
                'will be billed at the on-peak rate. All other hours are subject '
                'to the off-peak rate.'.format(
                    st.session_state.peak_hrs[0],
                    st.session_state.peak_hrs[1]
                )
            )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('### Electricity Price per kWh')
            if st.session_state.energy_rate_type == 'Time of Use (TOU)':
                price_cols = st.columns(2)
                price_cols[0].number_input(
                    'Peak Energy Cost ($/kWh)',
                    min_value=0., max_value=5., step=0.0001,
                    format='%.4f', key='peak_kwh_price'
                )
                price_cols[1].number_input(
                    'Off-Peak Energy Cost ($/kWh)',
                    min_value=0., max_value=5., step=0.0001,
                    format='%.4f', key='off_peak_kwh_price'
                )

            elif st.session_state.energy_rate_type == 'Constant':
                price_cols = st.columns(2)
                price_cols[0].number_input(
                    'Energy Cost ($/kWh)',
                    min_value=0., max_value=5., step=0.0001,
                    format='%.4f', key='constant_kwh_price'
                )

    with c2:
        with st.container(border=True):
            st.markdown('### Demand Charges per kW')
            kw_price_cols = st.columns(2)
            checkbox_cols = st.columns(2)
            with checkbox_cols[0]:
                st.checkbox(
                    'Include demand charge?',
                    key='rate_includes_demand_charge',
                )
            if st.session_state.rate_includes_demand_charge:
                if st.session_state.energy_rate_type == 'Time of Use (TOU)':
                    checkbox_cols[1].checkbox(
                        'TOU demand charges?',
                        key='tou_demand_charge'
                    )

                else:
                    st.session_state.tou_demand_charge = False

                if st.session_state.tou_demand_charge:
                    kw_price_cols[0].number_input(
                        'Peak Demand Charge ($/kW)',
                        min_value=0., max_value=100.,
                        key='peak_demand_charge'
                    )
                    kw_price_cols[1].number_input(
                        'Off-Peak Demand Charge ($/kW)',
                        min_value=0., max_value=100.,
                        key='off_peak_demand_charge'
                    )

                else:
                    kw_price_cols[0].number_input(
                        'Demand Charge ($/kW)',
                        min_value=0., max_value=100.,
                        key='demand_charge'
                    )

with load_tab:
    st.markdown('### Estimated Charging Load Profile')
    method_name = {
        'Charge for All Available Time': 'min_power',
        'Charge at Fixed Power': '60kW',
        'Prioritize Off-Peak Charging': 'off_peak'
    }

    # For each block, develop a charging profile for:
    # - overnight depot charging
    # - opportunity charging (if necessary)
    min_time_plot = st.session_state.beb_trips['start_time'].min()
    max_time_plot = st.session_state.beb_trips['start_time'].max() \
        + datetime.timedelta(days=1)
    if st.session_state.energy_rate_type == 'Time of Use (TOU)':
        peak_hour_start, peak_hour_end = st.session_state.peak_hrs
    else:
        peak_hour_start = None
        peak_hour_end = None

    depot_lp = get_depot_load_cached(
        date=st.session_state.date_set,
        routes=st.session_state.chosen_rts,
        kwh_per_mi=st.session_state.kwh_per_mi,
        battery_kwh=st.session_state.battery_kwh,
        method=method_name[st.session_state.charge_method],
        min_time=min_time_plot, max_time=max_time_plot,
        peak_start=peak_hour_start, peak_end=peak_hour_end
    )

    oppo_lp = get_oppo_load_cached(
        date=st.session_state.date_set,
        routes=st.session_state.chosen_rts,
        kwh_per_mi=st.session_state.kwh_per_mi,
        max_kwh=st.session_state.battery_kwh,
        min_time=min_time_plot, max_time=max_time_plot
    )

    st.session_state.load_profile = depot_lp + oppo_lp
    st.session_state.load_profile.rename('All Chargers')

    lp_choice = st.columns(3)[0].selectbox(
        'Select load profile to view',
        options=[
            'All charging', 'Depot charging only', 'Opportunity charging only'
        ]
    )
    plot_lp = {
        'All charging': st.session_state.load_profile,
        'Depot charging only': depot_lp,
        'Opportunity charging only': oppo_lp
    }[lp_choice]

    st.plotly_chart(plot_lp.plot('10min'))

    st.download_button(
        'Download load profile as CSV',
        data=plot_lp.to_csv(),
        file_name='zebra_load_profile.csv'
    )
    # st.plotly_chart(oppo_lp.plot('10min'))
    # st.plotly_chart(st.session_state.load_profile.plot('10min'))

with cost_tab:
    c1, c2, c3 = st.columns((1, 2, 1))
    c1.markdown('### Load Metrics')
    c3.markdown('### vs. Diesel Buses')

    # Calculate costs from composite load profile
    cost_df = st.session_state.load_profile.load_srs.to_frame()
    if st.session_state.energy_rate_type == 'Time of Use (TOU)':
        cost_df['hour'] = cost_df.index.hour
        cost_df['kwh_price'] = st.session_state.off_peak_kwh_price
        cost_df['peak'] = False
        peak_idx = (
            cost_df['hour'] >= peak_hour_start.hour
        ) & (
            cost_df['hour'] < peak_hour_end.hour
        )
        cost_df.loc[peak_idx, 'peak'] = True
        cost_df.loc[peak_idx, 'kwh_price'] = st.session_state.peak_kwh_price
        cost_df = cost_df.drop(columns=['hour'])

    # Add demand charges
    else:
        cost_df['kwh_price'] = st.session_state.constant_kwh_price

    if st.session_state.rate_includes_demand_charge:
        if st.session_state.tou_demand_charge:
            cost_df['kw_price'] = st.session_state.off_peak_demand_charge
            cost_df.loc[peak_idx, 'kw_price'] = \
                st.session_state.peak_demand_charge
        else:
            cost_df['kw_price'] = st.session_state.demand_charge
    else:
        cost_df['kw_price'] = 0

    cost_df['kwh_cost'] = cost_df['kwh_price'] * cost_df['power'] / 60
    cost_df['kw_cost'] = cost_df['kw_price'] * cost_df['power']

    c1.metric(
        'Total kWh', millify(cost_df['power'].sum() / 60),
        help='The total kilowatt-hours of electricity needed to fully recharge'
             ' all buses in your system on {}.'.format(
            st.session_state.date_set.strftime('%m/%d/%y')
        )
    )

    overall_kw_max = st.session_state.load_profile.max_kw()
    max_power_millify = millify(overall_kw_max * 1000, prefixes=['k', 'M', 'G']) + 'W'
    max_power_str = max_power_millify[:-2] + ' ' + max_power_millify[-2:]

    c1.metric(
        'Maximum Power Demand', max_power_str
    )

    total_kwh_cost = cost_df['kwh_cost'].sum()

    if st.session_state.rate_includes_demand_charge:
        monthly_demand_charge = cost_df['kw_cost'].max()
    else:
        monthly_demand_charge = 0

    # source: https://afdc.energy.gov/conserve/public-transportation
    # TODO: be careful about whether layover distance is being tracked
    diesel_mpg = 3.4
    diesel_usd_per_gal = 4.5
    n_buses = st.session_state.beb_trips['block_id'].nunique()
    block_df = pd.DataFrame(
        st.session_state.beb_trips.groupby('block_id')[
            ['total_dist', 'dh_dist']].sum())
    block_df['combined_dist'] = block_df['total_dist'] + block_df['dh_dist']
    diesel_total_cost = diesel_usd_per_gal * block_df[
        'combined_dist'].sum() / diesel_mpg
    total_beb_cost = total_kwh_cost + (monthly_demand_charge * 12 / 365.25)

    # Compile cost information
    cost_components = dict()
    if st.session_state.energy_rate_type == 'Time of Use (TOU)':
        peak_gb = cost_df.groupby('peak')
        totals_by_peak = peak_gb[['power', 'kwh_cost']].sum()
        c1.metric(
            'Peak Period kWh', millify(totals_by_peak.loc[True, 'power'] / 60)
        )

        c1.metric(
            'Off-Peak kWh', millify(totals_by_peak.loc[False, 'power'] / 60)
        )

        cost_components['Off-Peak Total kWh Cost'] = totals_by_peak.loc[
            False, 'kwh_cost']
        cost_components['Peak Period Total kWh Cost'] = totals_by_peak.loc[
            True, 'kwh_cost']

    else:
        cost_components['Total kWh Cost'] = cost_df['kwh_cost'].sum()

    cost_components['Normalized kW Cost per Day'] = \
        cost_df['kw_cost'].max() * 12 / 365.25

    cost_labels = list(cost_components.keys())
    cost_values = list(cost_components.values())
    fig = go.Figure(
        data=[
            go.Pie(
                labels=cost_labels, values=cost_values, hole=0.6,
                title={'text': '<b> Total:<br>$' + millify(total_beb_cost) + '</b>', 'font.size': 36},
            )
        ]
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="center",
            x=0.5
        )
    )
    fig.update_traces(
        hovertemplate="%{label}: $%{value:.2f}"
    )
    fig.update_layout(
        title={
            'text': 'Cost Breakdown',
            'xanchor': 'center',
            'x': 0.5,
            'font.size': 24
        }
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=70, b=20),
    )
    c2.plotly_chart(fig, use_container_width=True)

    totals_cols = st.columns(3)
    # c3.metric(
    #     'Total BEB Fuel Cost (per Day)', '$' + millify(total_beb_cost)
    # )
    c3.metric(
        'Diesel Fuel Cost', '$' + millify(diesel_total_cost)
    )

    comp_cols = st.columns(3)
    c3.metric(
        'Total Daily Savings', '$' + millify(
            diesel_total_cost - total_beb_cost
        )
    )
    c3.metric(
        'Daily Savings Per Bus', '$' + millify(
            (diesel_total_cost - total_beb_cost) / n_buses
        )
    )


with help_tab:
    st.markdown('## Energy Prices')
    st.markdown(
        """
        Different utilities throughout the US implement widely variable
        pricing schemes for electricity. ZEBRA can't currently capture all
        possible billing strategies, but we model two of the most common
        tools utilities use to promote grid-friendly demand patterns: 
        time-of-use (TOU) pricing and demand charges. Time-of-use rates
        vary the cost of energy per kilowatt-hour depending on demand,
        while demand charges impose a penalty for the maximum power
        demand in kilowatts during a billing period. The default options
        below correspond to the "High Demand" commercial customer rates
        offered by Seattle Public Utilities as of May 15, 2024. 
        """
    )


with home_tab:
    c1, c2, c3 = st.columns((1, 3, 0.1))
    c1.metric(
        'Total Electricity Cost', '$' + millify(total_beb_cost),
        delta=millify(total_beb_cost - diesel_total_cost),
        delta_color='inverse',
        help='The total electricity cost per day, including normalized demand charge. '
             'The difference shown is the comparison to estimated diesel bus cost.'
    )

    c1.metric(
        'Maximum Power', max_power_str
    )
    lf_start = st.session_state.date_set + datetime.timedelta(hours=7)
    lf_profile = st.session_state.load_profile.filter_datetime(
        min_dt=lf_start, max_dt=lf_start + datetime.timedelta(days=1)
    )
    c1.metric(
        'Load Factor',
        '{:.2f}'.format(lf_profile.load_factor()),
        help='We calculate the load factor based on a 24-hour window that starts and ends at 7 a.m.'
    )
    lp_fig = st.session_state.load_profile.plot('10min')
    lp_fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
    )
    c2.plotly_chart(
        lp_fig,
        use_container_width=True
    )
    # c3.metric('Test', 5)

# style_metric_cards(
#     border_color='#4b2e83',
#     border_left_color='#4b2e83'
# )

st.markdown(
        f"""
        <style>
            div[data-testid="stMetric"],
            div[data-testid="metric-container"] {{
                border: 1px solid #4b2e83;
                padding: 10px 10px 10px 20px;
                border-radius: 5px;
                border-left: 0.5rem solid #4b2e83 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
)

# st.markdown(
#         f"""
#         <style>
#             div[data-testid="stMetric"],
#             div[data-testid="metric-container"] {{
#                 margin: auto;
#             }}
#             [data-testid="metric-container"] > div {{
#                 margin: auto;
#             }}
#             [data-testid="metric-container"] label {{
#                 margin: auto;
#             }}
#         </style>
#         """,
#         unsafe_allow_html=True,
# )
