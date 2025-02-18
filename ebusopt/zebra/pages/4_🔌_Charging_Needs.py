import streamlit as st
import pandas as pd
import plotly.express as px
from ebusopt.zebra.utils import plot_trips_and_terminals, config_page,\
    read_markdown

# TODO: hide this page if no charging is necessary. Alternatively,
#   change the block exploring visuals to include all blocks

st.set_page_config(
    page_title='Charging Needs',
    page_icon='🦓',
    layout='wide'
)
config_page()


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

if st.session_state.routes_changed:
    # Add remaining fields to DF
    with st.spinner(
        'Calculating trip distances. Please wait...'
    ):
        st.session_state.beb_trips = st.session_state.gtfs.add_trip_data(
            st.session_state.beb_trips)
        st.session_state.routes_changed = False

if 'heatmap_range' not in st.session_state:
    st.session_state.heatmap_range = st.session_state.range_input

st.markdown(
    "<h2 style='text-align: center; color: black;'>Recharging Needs</h1>",
    unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(
    ['🪫 Hardest Routes', '🚌 Longest Blocks', '📍 Recharging Map', '❓ Help'])

with t1:
    block_df = pd.DataFrame(
            st.session_state.beb_trips.groupby('block_id')[['total_dist', 'dh_dist']].sum())
    block_df['combined_dist'] = block_df['total_dist'] + block_df['dh_dist']
    if 'lo_blocks' not in st.session_state:
        st.session_state.lo_blocks = block_df[
            block_df['combined_dist'] > st.session_state.range_input]

    st.markdown(
        'You specified a range of **:blue[{}]** miles. With the specified '
        'range, there are '
        '**:blue[{} blocks that require charging]** in order to be completed '
        'as currently scheduled. On this page, you can explore'
        ' more about these blocks that would require layover/opportunity charging '
        'or schedule adjustments in order to be served by BEBs with the specified '
        'range.'.format(
            st.session_state.range_input, len(st.session_state.lo_blocks)))

    st.markdown('### Which Routes Most Often Need Charging?')

    lo_trips = st.session_state.beb_trips[
        st.session_state.beb_trips['block_id'].isin(
            st.session_state.lo_blocks.index.tolist())]
    st.session_state.lo_trips = lo_trips

    lo_counts = lo_trips.groupby(
        'route_short_name')['block_id'].nunique().rename(
        'Number of Out-of-Range Blocks').to_frame()
    lo_counts['Number of Trips on Out-Of-Range Blocks'] = lo_trips.groupby(
        'route_short_name')['trip_id'].nunique()
    lo_counts.index = lo_counts.index.rename('Route Name')

    st.dataframe(
        lo_counts.sort_values(
            by='Number of Out-of-Range Blocks', ascending=False)
    )

with t2:
    st.markdown('### Which Blocks Require the Most Range?')
    blocks_by_dist = st.session_state.lo_blocks.sort_values(
        by='combined_dist', ascending=False)
    c1, c2 = st.columns([0.5, 0.5])
    chosen_block = c1.selectbox('Select a Block',
                                ['See all'] + blocks_by_dist.index.tolist())

    if chosen_block == 'See all':
        blocks_by_dist['Routes Served'] = st.session_state.beb_trips.groupby(
            'block_id')['route_short_name'].unique()
        st.dataframe(
            blocks_by_dist.reset_index(),
            column_config={
                'block_id': st.column_config.TextColumn('Block ID'),
                'combined_dist': st.column_config.NumberColumn(
                    'Total Distance (miles)', format='%d'
                ),
                'total_dist': st.column_config.NumberColumn(
                    'Distance in Service (miles)', format='%d'
                ),
                'dh_dist': st.column_config.NumberColumn(
                    'Deadhead Distance (miles)', format='%d'
                )
            },
            column_order=['block_id', 'combined_dist', 'total_dist', 'dh_dist'],
            hide_index=True,
            use_container_width=True
        )

    else:
        chosen_trips = lo_trips[lo_trips['block_id'] == chosen_block].sort_values(
            by='trip_idx'
        )
        chosen_rts = chosen_trips['route_short_name'].unique()

        if len(chosen_rts) > 1:
            rts_str = 'serves trips on routes {}'.format(chosen_rts[0])
            for r in chosen_rts[1:-1]:
                rts_str += ', {}'.format(r)

            rts_str += ' and {}.'.format(chosen_rts[-1])

        else:
            rts_str = 'serves trips on route {}.'.format(chosen_rts[0])

        st.markdown(
            'This block has a total distance of {:.0f} miles. '
            'It completes a total of {} trips during the day and {} '
            'The block is active from {} until {}.'.format(
                st.session_state.lo_blocks.loc[chosen_block, 'combined_dist'],
                len(chosen_trips),
                rts_str,
                chosen_trips['start_time'].min(),
                chosen_trips['end_time'].max()
            )
        )

        cols = st.columns(2)
        if cols[0].button('Map this block'):
            fig = plot_trips_and_terminals(
                trips_df=st.session_state.beb_trips[
                    st.session_state.beb_trips['block_id'] == chosen_block],
                locs_df=None,
                shapes_df=st.session_state.gtfs.shapes_df
            )
            st.plotly_chart(fig)

        if cols[1].button('See all block data'):
            st.dataframe(chosen_trips)

with t3:
    c1, c2 = st.columns(2)

    if 'lo_trips' not in st.session_state:
        st.session_state.lo_trips = st.session_state.beb_trips[
            st.session_state.beb_trips['block_id'].isin(
                st.session_state.lo_blocks.index.tolist())]
    layover_trips_df = st.session_state.lo_trips

    # map_choice = c1.selectbox(
    #     'Select map type', ['Heatmap', 'Scatter'])
    map_choice = 'Heatmap'

    with st.columns((0.8, 0.2))[0]:
        map_container = st.container()
        st.slider(
            'Bus Range (miles)', min_value=50,
            max_value=int(block_df['combined_dist'].max()), key='heatmap_range',
            help='Adjust the slider to see how range impacts potential '
                 'opportunity charging sites. Note that this input only '
                 'impacts the above heatmap and does not change the range '
                 'value set elsewhere in the zebra.')
    lo_blocks_heatmap = block_df[
        block_df['combined_dist'] > st.session_state.heatmap_range]
    lo_trips_heatmap = st.session_state.beb_trips[
        st.session_state.beb_trips['block_id'].isin(
            lo_blocks_heatmap.index.tolist())]

    if map_choice == 'Heatmap':
        with map_container:
            start_cts = lo_trips_heatmap.groupby(
                ['start_lat', 'start_lon']).count()['route_id'].rename('start')
            start_cts.index.set_names(['lat', 'lon'], inplace=True)
            end_cts = lo_trips_heatmap.groupby(
                ['end_lat', 'end_lon']).count()['route_id'].rename('end')
            end_cts.index.set_names(['lat', 'lon'], inplace=True)
            all_cts = pd.merge(
                start_cts, end_cts, left_index=True, right_index=True, how='outer')
            all_cts = all_cts.fillna(0)
            all_cts['total'] = all_cts['start'] + all_cts['end']
            all_cts = all_cts.sort_values(by='total',
                                          ascending=False).reset_index()
            all_cts = all_cts.astype(
                {
                    'start': int,
                    'end': int,
                    'total': int
                }
            )
            all_cts = all_cts.rename(
                columns={
                    'lat': 'Latitude',
                    'lon': 'Longitude',
                    'start': 'Number of Trips Starting Here',
                    'end': 'Number of Trips Ending Here',
                    'total': 'Total Number of Trips'
                }
            )

            if 'heatmap_radius' not in st.session_state:
                st.session_state.heatmap_radius = 20

            fig = px.density_mapbox(
                all_cts, lat='Latitude', lon='Longitude', z='Total Number of Trips',
                mapbox_style='carto-positron',
                radius=st.session_state.heatmap_radius,
                title='Heatmap of Trip Terminals for Charging Blocks')
            fig.update_layout(
                margin={'b': 0, 't': 50},
            )
            st.plotly_chart(fig, use_container_width=True)

        # TODO: cache this csv conversion
        st.download_button(
            'Download current heatmap data',
            data=all_cts.to_csv(index=False),
            file_name='zebra_heatmap_data.csv'
        )

    else:
        include_routes = st.checkbox(
            'Overlay trip shapes?', value=False,
            help='This may slow map rendering if there are many distinct '
                 'shapes to plot.'
        )

        if include_routes:
            shapes_df = st.session_state.gtfs.shapes_df
        else:
            shapes_df = None

        inst_map = plot_trips_and_terminals(
            trips_df=lo_trips_heatmap, locs_df=None,
            shapes_df=shapes_df).update_layout(
                margin={
                    't': 0, 'b': 0
                }
        )
        st.plotly_chart(inst_map, use_container_width=True)


with t4:
    st.markdown(read_markdown('ebusopt/zebra/text/charging_needs_help.md'))

