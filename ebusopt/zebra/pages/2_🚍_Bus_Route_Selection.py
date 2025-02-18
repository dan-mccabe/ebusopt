import streamlit as st
import pandas as pd
from ebusopt.zebra.utils import config_page, read_markdown


def set_change_flag():
    st.session_state.routes_changed = True
    st.session_state.beb_trips = st.session_state.day_trips[
        st.session_state.day_trips['route_short_name'].isin(
            st.session_state.chosen_rts)
    ]


st.set_page_config(
    page_title='Bus Routes to Include',
    page_icon='🦓',
    layout='wide')

config_page()

if 'gtfs_run' not in st.session_state:
    st.error('Please specify GTFS data before visiting this screen.')
    st.stop()

if not st.session_state.gtfs_run:
    st.error('Please specify GTFS data before visiting this screen.')
    st.stop()

st.markdown(
    "<h2 style='text-align: center; color: black;'>Select Bus Routes</h1>",
    unsafe_allow_html=True)

t1, t2 = st.tabs(['🖱️ Select Routes', '❓ Help'])


with t1:
    # Build up list of route names, sorting strings and numbers
    rt_names = st.session_state.day_trips['route_short_name'].unique()
    numeric_routes = pd.to_numeric(rt_names, errors='coerce')
    # Get and sort names that aren't numeric
    str_names = sorted(rt_names[pd.isna(numeric_routes)].tolist())
    # Get and sort names that are numeric, then turn back to strings
    num_names = sorted(numeric_routes[~pd.isna(numeric_routes)].astype(int).tolist())
    num_names = [str(i) for i in num_names]
    day_rts = num_names + str_names

    # Set default routes to choose
    if 'chosen_rts' not in st.session_state:
        st.session_state.chosen_rts = day_rts
    else:
        if any(i not in day_rts for i in st.session_state.chosen_rts):
            st.session_state.chosen_rts = day_rts

    st.multiselect(
        'Select Bus Routes', day_rts, key='chosen_rts',
        on_change=set_change_flag
    )

    if st.session_state.gtfs_changed:
        # Streamlit does not run the on_change callback above the very
        # first time that the route selection widget is loaded with all
        # the routes. So, beb_trips is undefined. As a workaround, we
        # can use the gtfs_changed flag set on Page 1.
        set_change_flag()
        st.session_state.gtfs_changed = False

    st.markdown(
        'Buses on these routes serve **:blue[{} trips]** on'
        ' **:blue[{} blocks]** on your chosen day.'.format(
            len(st.session_state.beb_trips),
            len(st.session_state.beb_trips['block_id'].unique())
        ))

with t2:
    st.markdown(read_markdown('ebusopt/zebra/text/route_selection_help.md'))
