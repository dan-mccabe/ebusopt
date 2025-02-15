import streamlit as st
from ebusopt.zebra.utils import config_page

st.set_page_config(
    page_title='Energy Costs',
    page_icon='🦓',
    layout='wide'
)
config_page()
st.markdown(
    "<h2 style='text-align: center; color: black;'>Adjust Parameters</h2>",
    unsafe_allow_html=True
)


def update_energy_params(changed_param):
    if changed_param == 'kwh_per_mi':
        st.session_state.range_input = st.session_state.battery_kwh / st.session_state.kwh_per_mi

    elif changed_param == 'battery_size':
        st.session_state.range_input = st.session_state.battery_kwh / st.session_state.kwh_per_mi

    elif changed_param == 'range':
        st.session_state.battery_kwh = st.session_state.range_input * st.session_state.kwh_per_mi

    else:
        raise ValueError('invalid changed_param value')


with st.expander('BEB Range and Energy Consumption', True):
    range_cols = st.columns(3)
    with range_cols[0]:
        st.number_input(
            'BEB Range (Miles)', min_value=0, max_value=1000,
            key='range_input', on_change=update_energy_params, args=['range'])

    with range_cols[1]:
        st.number_input(
            'Energy Consumption Rate (kWh/mi)',
            min_value=0., max_value=20., step=0.01, key='kwh_per_mi',
            on_change=update_energy_params, args=['kwh_per_mi']
        )
        # st.session_state.beb_trips['kwh_per_mi'] = st.session_state.kwh_per_mi

    with range_cols[2]:
        st.number_input(
            'Battery Size (kWh)',
            min_value=20, max_value=2000, key='battery_kwh',
            on_change=update_energy_params, args=['battery_size']
        )


with st.expander('Recharging Strategy', True):
    method_options = ['Charge for All Available Time', 'Charge at Fixed Power']
    if st.session_state.energy_rate_type == 'Time of Use (TOU)':
        method_options.append('Prioritize Off-Peak Charging')
    with st.columns(3)[0]:
        st.selectbox(
            'Charging Method',
            options=method_options,
            key='charge_method'
        )