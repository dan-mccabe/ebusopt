import streamlit as st
import plotly.express as px
import pandas as pd
from ebusopt.zebra.utils import config_page, read_markdown

st.set_page_config(
    page_title='Understand Your Range Requirements',
    page_icon='🦓',
    layout='wide'
)
config_page()
st.markdown(
    "<h2 style='text-align: center; color: black;'>Range Requirements</h1>",
    unsafe_allow_html=True)


if 'gtfs_run' not in st.session_state:
    st.error('Please specify GTFS data before visiting this screen.')
    st.stop()

if not st.session_state.gtfs_run:
    st.error('Please specify GTFS data before visiting this screen.')
    st.stop()

if 'beb_trips' not in st.session_state:
    st.session_state.beb_trips = st.session_state.day_trips

if st.session_state.routes_changed:
    # Add remaining fields to DF
    with st.spinner(
        'Calculating trip distances. Please wait...'
    ):
        st.session_state.beb_trips = st.session_state.gtfs.add_trip_data(
            st.session_state.beb_trips, st.session_state.date_set
        )
        st.session_state.beb_trips = st.session_state.gtfs.add_deadhead(
            st.session_state.beb_trips
        )
        st.session_state.routes_changed = False
block_df = pd.DataFrame(
    st.session_state.beb_trips.groupby('block_id')[
        ['total_dist', 'dh_dist']].sum()
)
block_df['combined_dist'] = block_df['total_dist'] + block_df['dh_dist']


if 'range_input' not in st.session_state:
    st.session_state.range_input = 150
st.session_state.range_set = True

depot_blocks = block_df[block_df['combined_dist'] <= st.session_state.range_input]
lo_blocks = block_df[block_df['combined_dist'] > st.session_state.range_input]
st.session_state.lo_blocks = lo_blocks

t1, t2, t3 = st.tabs(['📊 Block Distances', '🔋 Range Requirements', '❓ Help'])

with t1:
    c1c, c2c = st.columns(2)

    with c1c:
        st.plotly_chart(
            px.histogram(
                block_df, 'combined_dist',
                title='Histogram of Block Distances',
                labels={
                    'combined_dist': 'Distance (miles)',
                    'count': 'Number of Blocks'
                }
            ).update_layout(yaxis_title='Number of Blocks'),
            use_container_width=True
        )

        export_df = block_df.rename(
            columns={
                'block_id': 'Block ID',
                'total_dist': 'Distance in Service (miles)',
                'dh_dist': 'Deadhead Distance (miles)',
                'combined_dist': 'Total Distance (miles)'
            }
        )
        st.download_button(
            'Download Block Distance Data',
            data=export_df.to_csv(),
            file_name='zebra_block_distances.csv'
        )

    # else:
    with c2c:
        # Note: add marginal='histogram' kwarg to px.ecdf() to get
        # both on one plot if desired
        st.plotly_chart(
            px.ecdf(
                block_df, 'combined_dist',
                title='Distribution of Block Distances',
                labels={
                    'combined_dist': 'Distance (miles)',
                    'count': 'Number of Blocks'
                },
                ecdfnorm='percent').update_layout(
                    yaxis_title='Percentage of Blocks Covered'
            ),
            use_container_width=True
        )

with t2:

    c1, c2 = st.columns(2)
    c1.markdown(
        'With the specified range, depot charging alone is enough to complete '
        '**:blue[{:.1f}% of blocks ({:,} out of {:,})]** '.format(
            100*len(depot_blocks) / len(block_df), len(depot_blocks),
            len(block_df))
    )

    c2.markdown(
        'These blocks account for **:blue[{:,} out of {:,} ({:.1f}%) total'
        ' daily vehicle miles.]** This distance consists of {:,} miles in '
        'passenger service and an estimated {:,} miles of deadhead.'.format(
            int(depot_blocks['combined_dist'].sum()),
            int(block_df['combined_dist'].sum()),
            100 * depot_blocks['combined_dist'].sum()
            / block_df['combined_dist'].sum(),
            int(block_df['total_dist'].sum()),
            int(block_df['dh_dist'].sum()))
    )

    if len(lo_blocks) > 0:
        range_deficit = lo_blocks['combined_dist'] - st.session_state.range_input
        # st.plotly_chart(
        #     px.histogram(
        #         range_deficit, 'combined_dist',
        #         title='Histogram of Range Deficit for Layover Blocks',
        #         labels={
        #             'combined_dist': 'Range Deficit (miles)',
        #             'count': 'Number of Blocks'
        #         }
        #     ).update_layout(yaxis_title='Number of Blocks')
        # )

        st.plotly_chart(
            px.ecdf(
                range_deficit, 'combined_dist',
                title='Distribution of Range Deficit for Charging Blocks',
                labels={
                    'combined_dist': 'Range Deficit (miles)'
                },
                marginal='histogram',
                ecdfnorm='percent',
            ).update_layout(
                yaxis_title='Percentage of Blocks'),
            use_container_width=True
        )

        st.markdown(
            'If current bus schedules are retained, opportunity charging at stops '
            'or layover sites would need to extend the range of all buses in the '
            'system by a total of **:blue[{:,} miles]**, an average of **:blue[{:,} miles per bus.]**'
            ''.format(int(range_deficit.sum()), int(range_deficit.sum() / len(
                lo_blocks)
            ))
        )

with t3:
    help_md = read_markdown('ebusopt/zebra/text/range_requirements_help.md')
    st.markdown(help_md)

