import streamlit as st
import pandas as pd
from millify import millify
from ebusopt.zebra.utils import config_page, read_markdown

st.set_page_config(
    page_title='Emissions Impact',
    page_icon='🦓',
    layout='wide'
)
config_page()
st.markdown(
    "<h2 style='text-align: center; color: black;'>Emissions Impacts</h2>",
    unsafe_allow_html=True)

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

x = 10
# Calculate VMT across all trips, depot blocks, and layover blocks
# TODO: we seem to get different results if we sum over trips directly
#   vs. over blocks
block_df = pd.DataFrame(
    st.session_state.beb_trips.groupby('block_id')['total_dist'].sum())
depot_blocks = block_df[block_df['total_dist'] <= st.session_state.range_input]
lo_blocks = block_df[block_df['total_dist'] > st.session_state.range_input]
total_vmt = block_df['total_dist'].sum()
depot_vmt = depot_blocks['total_dist'].sum()
lo_vmt = lo_blocks['total_dist'].sum()
# total_vmt = st.session_state.beb_trips['total_dist'].sum()
# depot_trips = st.session_state.beb_trips[
#     ~st.session_state.beb_trips['trip_id'].isin(
#         st.session_state.lo_trips['trip_id'])]
# depot_vmt = depot_trips['total_dist'].sum()
# lo_vmt = st.session_state.lo_trips['total_dist'].sum()

t1, t2 = st.tabs(['🌱 Results', '❓ Help'])

with t1:
    st.markdown('### Total Vehicle Miles Traveled')
    c1, c2, c3 = st.columns(3)
    c1.metric(label='Total Daily VMT', value=millify(total_vmt))
    c2.metric(label='VMT of In-Range Blocks', value=millify(depot_vmt))
    c3.metric(label='VMT of Out-of-Range Blocks', value=millify(lo_vmt))

    cnv_fct = (3.05 - 0.97) / 1000
    co2_saved = total_vmt * cnv_fct
    depot_saved = depot_vmt * cnv_fct
    lo_saved = lo_vmt * cnv_fct

    st.markdown('----')
    st.markdown('### Greenhouse Gas Emissions')
    c1, c2, c3 = st.columns(3)
    c1.metric(label='Total CO2 savings (metric tons)', value=millify(co2_saved))
    c2.metric(label='CO2 savings (in-range blocks)', value=millify(depot_saved))
    c3.metric(label='CO2 savings (fast-charging blocks)', value=millify(lo_saved))

with t2:
    st.markdown(read_markdown('ebusopt/zebra/text/emissions_help.md'))
