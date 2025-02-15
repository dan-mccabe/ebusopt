from ebusopt.gtfs_beb import get_shape
from pathlib import Path
from dotenv import dotenv_values, find_dotenv
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def read_markdown(filepath):
    return Path(filepath).read_text()


def config_page():
    css = '''
        <style>
            section.main > div {max-width:65rem}
        </style>
        '''
    st.markdown(css, unsafe_allow_html=True)

    st.markdown("""
            <style>
                   .block-container {
                        padding-top: 1rem;
                        padding-bottom: 5rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
            </style>
            """, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
        """,
        unsafe_allow_html=True)

    for k, v in st.session_state.items():
        st.session_state[k] = v


def plot_trips_and_terminals(
        trips_df: pd.DataFrame, shapes_df: pd.DataFrame, locs_df=None):
    # Mapbox API key
    try:
        config = dotenv_values(find_dotenv())
        token = config['MAPBOX_KEY']

    except KeyError:
        raise KeyError(
            'No Openrouteservice key found in .env file. For more information'
            ', please see the project README file.'
        )

    px.set_mapbox_access_token(token)

    # Compile terminal counts
    start_cts = trips_df.groupby(
        ['start_lat', 'start_lon']).count()['route_id'].rename('start')
    start_cts.index.set_names(['lat', 'lon'], inplace=True)
    end_cts = trips_df.groupby(
        ['end_lat', 'end_lon']).count()['route_id'].rename('end')
    end_cts.index.set_names(['lat', 'lon'], inplace=True)
    all_cts = pd.merge(
        start_cts, end_cts, left_index=True, right_index=True, how='outer')
    all_cts = all_cts.fillna(0)
    all_cts['total'] = all_cts['start'] + all_cts['end']
    all_cts = all_cts.sort_values(by='total', ascending=False).reset_index()
    all_cts['name'] = ''
    all_cts['symbol'] = 'circle'
    all_cts['size'] = all_cts['total']
    all_cts['label_name'] = [
        '{} trips start here, {} trips end here'.format(
            int(all_cts['start'][i]), int(all_cts['end'][i]))
        for i in range(len(all_cts))]
    all_cts['color'] = 'blue'

    # Trip terminals
    # Marker size: scale linearly from minimum to maximum
    min_marker = 10
    max_marker = 20
    msize = np.round(min_marker + (max_marker - min_marker) * (
            all_cts['size'] - all_cts['size'].min())/all_cts['size'].max())

    # Initialize figure
    # TODO: do this without re-plotting points
    fig = px.scatter_mapbox(
            all_cts.iloc[:1], lat='lat', lon='lon')

    new_trace = go.Scattermapbox(lat=all_cts['lat'], lon=all_cts['lon'],
                                 showlegend=True, hoverinfo='text',
                                 mode='markers', text=all_cts['label_name'],
                                 marker=go.scattermapbox.Marker(
                                     color='rgba(60, 120, 255, 1)',
                                     size=msize),
                                 name='Trip Start/End Locations ')
    fig.add_trace(new_trace)

    # Trips
    # TODO: figure renders super slowly. probably because we have so
    #   many trips to plot and plot every single point.
    shape_cts = trips_df.groupby('shape_id').count()['route_id'].sort_values()
    # st.write('{} shapes to plot'.format(len(shape_cts)))
    # st.write('{} unique routes'.format(len(trips_df['route_id'].unique())))

    # shapes_gb = shapes_df.sort_values(
    #     by='shape_pt_sequence').groupby('shape_id')
    # gdf_dict = {'shape_id': list(), 'geometry': list()}

    # for _, shape in shapes_gb:
    #     gdf_dict["shape_id"].append(shape['shape_id'])
    #     gdf_dict["geometry"].append(
    #         LineString(list(zip(
    #             shape['shape_pt_lon'], shape['shape_pt_lat']))))
    #
    # shapes_geo = gpd.GeoDataFrame(gdf_dict)

    if shapes_df is not None:
        # TODO: simplify geometry somehow so we can plot more
        for shp in shape_cts.index:
            shape_lon, shape_lat = get_shape(shapes_df, shp)
            shape_pt_df = pd.DataFrame({'lon': shape_lon, 'lat': shape_lat})
            alpha = 0.2 + 0.3 * shape_cts[shp] / max(shape_cts)
            rgba_str = 'rgba(255, 80, 80, {:.2f})'.format(alpha)

            new_trace = go.Scattermapbox(mode='lines',
                                         lat=shape_pt_df["lat"],
                                         lon=shape_pt_df["lon"],
                                         showlegend=False, hoverinfo='skip',
                                         line={'color': rgba_str, 'width': 2})
            fig.add_trace(new_trace)

        # Trace for legend
        new_trace = go.Scattermapbox(
            mode='lines', lat=shape_pt_df["lat"],
            lon=shape_pt_df["lon"], showlegend=True,
            line={'color': 'rgba(255, 80, 80, 0.9)'},
            name='Passenger Trip   ')
        fig.add_trace(new_trace)

    # Charging sites
    if locs_df is not None:
        locs_df = locs_df.set_index('name')
        locs_df['symbol'] = 'fuel'
        fig = px.scatter_mapbox(
            locs_df, lat='y', lon='x', text='label_name', zoom=9, size_max=24,
            hover_data={c: False for c in locs_df.columns})

        fig.update_traces(marker={'size': 10, 'symbol': locs_df['symbol']})
        fig.update_traces(textposition='bottom center', textfont_size=15,
                          textfont_color='white')

    # Reverse order to put markers on top
    fdata = fig.data
    fig.data = tuple(list(fdata[1:]) + [fdata[0]])
    fig = fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=0.98, font={'size': 14}),
        margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(mapbox_style="carto-positron")

    return fig

