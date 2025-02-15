import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polyline
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import logging
import pickle
from matplotlib.dates import HourLocator, DateFormatter
from matplotlib import ticker
from pathlib import Path
import datetime
from ebusopt.gtfs_beb import get_gmap_directions, GTFSData, get_shape
from dotenv import dotenv_values, find_dotenv

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times",
    "font.serif": "Times",
    "font.sans-serif": ["Times"],
    'font.size': 16})
sns.set_theme(font_scale=2)
sns.set_style({
    "text.usetex": False,
    "font.family": "Times",
    "font.serif": "Times",
    "font.sans-serif": ["Times"],
    'font.size': 16,
    'axes.grid': False,
    'axes.ticks': True})
# sns.set_context('paper')
# sns.axes_style('ticks')


def plot_bus_soc(
        result_df: pd.DataFrame, block_id: int, capacity=466,
        soc_lb=0.2, soc_ub=0.95):
    """
    Plot the battery level of a bus over the course of the day.
    :param result_df: pandas DataFrame of model results. Needs columns
        block_id, trip_idx, soc
    :param block_id: block ID of bus
    :param soc_lb: minimum state of charge allowed
    :param soc_ub: maximum state of charge allowed
    :param capacity: battery capacity in kWh (assume usable from 20% to 95%)
    """
    block_df = result_df[result_df['block_id'] == block_id]
    chg_srs = block_df.sort_values(by='trip_idx')['soc'].reset_index(drop=True)
    chg_srs += soc_lb*capacity
    idx = chg_srs.index.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(idx, chg_srs.to_numpy(), label='Battery Level')
    ax.plot(idx, soc_lb*capacity*np.ones(len(chg_srs)),
            label='Min. Charge')
    ax.plot(idx, soc_ub*capacity*np.ones(len(chg_srs)),
            label='Max. Charge')
    y_upper = 100 * (int(soc_ub*capacity / 100) + 1)
    ax.set_ylim([-10, y_upper])
    ax.set_xlabel('Trip Index')
    ax.set_ylabel('Battery Charge (kWh)')
    ax.set_title('Block ID: ' + str(block_id))
    ax.legend()
    return fig


def plot_energy_needs(flm):
    flm.check_charging_needs()
    flm.check_feasibility()

    # Histogram of block energy needs
    energy_bins = range(0, 950, 50)
    fig = plt.figure()
    plt.hist(flm.block_energy_needs.values(), bins=energy_bins,
             label='Non-Charging Blocks')
    plt.title('Histogram of Block Energy Needs')
    plt.xlabel('Energy Required to Complete Block (kWh)')
    plt.ylabel('Number of Blocks')

    # Limit to charging blocks
    charging_block_needs = [
        flm.block_energy_needs[v] for v in flm.charging_vehs]
    plt.hist(charging_block_needs, bins=energy_bins, label='Charging Blocks')
    plt.legend()
    return fig


def plot_layover_times(flm):
    flm.check_charging_needs()
    flm.check_feasibility()

    # Histogram of recovery time
    recov_times = list()
    charging_recov = list()
    for (v, t) in flm.veh_trip_pairs:
        if t > 0:
            try:
                recov = flm.trip_start_times[v, t+1] - flm.trip_end_times[v, t]
                recov_times.append(recov)
                if v in flm.charging_vehs:
                    charging_recov.append(recov)
            except KeyError:
                pass

    recov_bins = range(0, 105, 5)
    fig = plt.figure()
    plt.hist(recov_times, bins=recov_bins, label='Non-Charging Blocks')
    plt.title('Histogram of Scheduled Layover Times')
    plt.xlabel('Layover Time (min.)')
    plt.ylabel('Number of Trips')

    # Limit to charging blocks
    plt.hist(charging_recov, bins=recov_bins, label='Charging Blocks')
    plt.legend()
    return fig


def plot_conflict_sets(flm):
    flm.check_charging_needs()
    flm.check_feasibility()
    flm.set_conflict_sets()

    conflict_set_sizes = {vts: len(flm.conflict_sets[vts])
                          for vts in flm.conflict_sets}

    fig = plt.figure()
    max_conflict = max(conflict_set_sizes.values())
    plt.hist(conflict_set_sizes.values(), bins=range(0, max_conflict))
    plt.title('Histogram of Conflict Set Sizes')
    plt.xlabel('Number of Potentially Conflicting Vehicles')
    plt.xticks(range(0, max_conflict), range(0, max_conflict))
    plt.ylabel('Number of Occurrences')

    mean_cflt_sz = sum(conflict_set_sizes.values()) / len(
        conflict_set_sizes.values())
    logging.info('Mean conflict set size: {:.2f}'.format(mean_cflt_sz))

    return fig


def plot_trips_and_terminals(
        trips_df: pd.DataFrame, locs_df: pd.DataFrame,
        shapes_df: pd.DataFrame, light_or_dark: str = 'light'):
    # Mapbox API key
    try:
        config = dotenv_values(find_dotenv())
        token = config['MAPBOX_KEY']

    except KeyError:
        raise KeyError(
            'No Openrouteservice key found in .env file. For more information'
            ', please see the project README file.'
        )

    if light_or_dark == 'light':
        text_and_marker_color = 'black'

    elif light_or_dark == 'dark':
        text_and_marker_color = 'white'

    else:
        raise ValueError(f'Unrecognized light_or_dark value: {light_or_dark}')

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

    # Charging sites
    if locs_df is not None:
        # locs_df = locs_df.set_index('name')
        locs_df['symbol'] = 'fuel'
        fig = px.scatter_mapbox(
            locs_df, lat='lat', lon='lon', text='label_name', zoom=10,
            size_max=30, hover_data={c: False for c in locs_df.columns})

        fig.update_traces(marker={'size': 10, 'symbol': locs_df['symbol']})
        fig.update_traces(textposition='bottom center', textfont_size=15,
                          textfont_color=text_and_marker_color)

    else:
        fig = go.Figure()

    # Trip terminals
    # Marker size: scale linearly from minimum to maximum
    min_marker = 15
    max_marker = 30
    msize = np.round(min_marker + (max_marker - min_marker) * (
            all_cts['size'] - all_cts['size'].min())/all_cts['size'].max())

    new_trace = go.Scattermapbox(lat=all_cts['lat'], lon=all_cts['lon'],
                                 showlegend=True, hoverinfo='text',
                                 mode='markers', text=all_cts['label_name'],
                                 marker=go.scattermapbox.Marker(
                                     color='rgba(60, 120, 255, 1)',
                                     size=msize),
                                 name='Trip Start/End Locations    ')
    fig.add_trace(new_trace)

    # Trips
    shape_cts = trips_df.groupby('shape_id').count()['route_id'].sort_values()
    for shp in shape_cts.index:
        shape_pts = get_shape(shapes_df, shp)
        shape_pt_df = pd.DataFrame(shape_pts).transpose()
        alpha = 0.2 + 0.5 * shape_cts[shp] / max(shape_cts)
        rgba_str = 'rgba(255, 80, 80, {:.2f})'.format(alpha)

        new_trace = go.Scattermapbox(mode='lines',
                                     lat=shape_pt_df["shape_pt_lat"],
                                     lon=shape_pt_df["shape_pt_lon"],
                                     showlegend=False, hoverinfo='skip',
                                     line={'color': rgba_str, 'width': 2})
        fig.add_trace(new_trace)

    # Trace for legend
    new_trace = go.Scattermapbox(
        mode='lines', lat=shape_pt_df["shape_pt_lat"],
        lon=shape_pt_df["shape_pt_lat"], showlegend=True,
        line={'color': 'rgba(255, 80, 80, 0.9)'},
        name='Passenger Trip   ')
    fig.add_trace(new_trace)

    # Reverse order to put markers on top
    fdata = fig.data
    fig.data = tuple(list(fdata[1:]) + [fdata[0]])
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=0.98, font={'size': 14},
        ),
        margin=dict(l=0, r=0, t=5, b=0))

    # if light_or_dark == 'dark':
    fig.update_layout(mapbox_style=light_or_dark)

    return fig


def plot_one_trip(result_df: pd.DataFrame, loc_df: pd.DataFrame,
                  gtfs: GTFSData, v: int, t: int, depot_coords: tuple):
    """
    Plot a single trip using plotly. Will show trip and any deadheading
    that follows.

    :param result_df:
    :param loc_df:
    :param gtfs:
    :param v:
    :param t:
    :return:
    """

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

    result_df = result_df.set_index(['block_id', 'trip_idx'])
    trip_id = result_df.at[(v, t), 'trip_id']
    used_sites = result_df['chg_site'].dropna().unique().tolist()
    loc_df = loc_df[loc_df.index.isin(used_sites)]
    block_df = gtfs.get_trips_in_block(v)

    # Base map: charger locations
    fig1 = px.scatter_mapbox(
        loc_df, lat="y", lon="x", text='label_name', zoom=9, size_max=24,
        hover_data={c: False for c in loc_df.columns})
    fig1.update_traces(marker={'size': 10, 'symbol': loc_df['symbol']})
    fig1.update_traces(textposition='bottom center', textfont_size=18)
    fig2 = px.scatter_mapbox(
        loc_df, lat="y", lon="x", text='label_name', zoom=9, size_max=24,
        hover_data={c: False for c in loc_df.columns})
    fig2.update_traces(marker={'size': 10, 'symbol': loc_df['symbol']})
    fig2.update_traces(textposition='bottom center', textfont_size=18)
    st.text(trip_id)

    od_directions = dict()
    # Add trip terminals
    if trip_id == 0:
        # Starts at depot
        start_coords = depot_coords
        next_tid = result_df.at[(v, t+1), 'trip_id']
        end_coords = (
            block_df.at[next_tid, 'start_lat'],
            block_df.at[next_tid, 'start_lon'])
        next_start_coords = end_coords
    elif trip_id == 100:
        prev_tid = result_df.at[(v, t-1), 'trip_id']
        start_coords = (
            block_df.at[prev_tid, 'end_lat'],
            block_df.at[prev_tid, 'end_lon'])
        end_coords = depot_coords
        next_start_coords = None
    else:
        start_coords = (
            block_df.at[trip_id, 'start_lat'],
            block_df.at[trip_id, 'start_lon']
        )
        end_coords = (
            block_df.at[trip_id, 'end_lat'],
            block_df.at[trip_id, 'end_lon']
        )
        next_tid = result_df.at[(v, t+1), 'trip_id']
        if next_tid == 100:
            next_start_coords = end_coords
        else:
            next_start_coords = (
                block_df.at[next_tid, 'start_lat'],
                block_df.at[next_tid, 'start_lon']
            )

    # Plot trip
    if trip_id not in [0, 100]:
        shp = block_df.at[trip_id, 'shape_id']
        shape_pts = get_shape(gtfs.shapes_df, shp)
        shape_pt_df = pd.DataFrame(shape_pts).transpose()

        new_trace = go.Scattermapbox(
            mode='lines', lat=shape_pt_df["shape_pt_lat"],
            lon=shape_pt_df["shape_pt_lon"], showlegend=False,
            hoverinfo='text', text='Current trip', line={'color': 'green'})
        fig1.add_trace(new_trace)

    else:
        # Depot trip, need google maps directions
        try:
            directions = od_directions[start_coords, end_coords]
        except KeyError:
            directions = get_gmap_directions(
                start_coords, end_coords)
            od_directions[start_coords, end_coords] = directions

        if t == 0:
            text_label = 'Deadhead from depot to first trip start'
        else:
            text_label = 'Deadhead from last trip end to depot'

        plot_line = polyline.decode(directions)
        lat = [pt[0] for pt in plot_line]
        lon = [pt[1] for pt in plot_line]
        new_trace = go.Scattermapbox(
            mode='lines', lat=lat, lon=lon, showlegend=False,
            hoverinfo='text', text=text_label, line={'color': 'green'})
        fig1.add_trace(new_trace)

    # Plot deadhead
    chg_site = result_df.at[(v, t), 'chg_site']
    if pd.isna(chg_site):
        # No charging after this trip
        # Plot deadhead to next trip
        if next_start_coords:
            if end_coords != next_start_coords:
                try:
                    directions = od_directions[end_coords, next_start_coords]
                except KeyError:
                    directions = get_gmap_directions(
                        end_coords, next_start_coords)
                    od_directions[end_coords, next_start_coords] = directions

                plot_line = polyline.decode(directions)
                lat = [pt[0] for pt in plot_line]
                lon = [pt[1] for pt in plot_line]
                new_trace = go.Scattermapbox(
                    mode='lines', lat=lat, lon=lon, showlegend=False,
                    hoverinfo='text', text='Deadhead to next trip',
                    line={'color': 'yellow'})
                fig2.add_trace(new_trace)

    else:
        # Visit a charger after this trip. Plot deadhead to and from.
        # First DH trip: terminal to charger
        orig1 = end_coords
        dest1 = (loc_df.at[chg_site, 'y'], loc_df.at[chg_site, 'x'])

        try:
            directions1 = od_directions[orig1, dest1]
        except KeyError:
            directions1 = get_gmap_directions(orig1, dest1)
            od_directions[orig1, dest1] = directions1

        plot_line = polyline.decode(directions1)
        lat = [pt[0] for pt in plot_line]
        lon = [pt[1] for pt in plot_line]
        # trip_text = 'trips' if r['n_trips'] > 1 else 'trip'
        # text = '{:d} {} to {}'.format(r['n_trips'], trip_text, r['chg_site'])
        new_trace = go.Scattermapbox(
            mode='lines', lat=lat, lon=lon, showlegend=False,
            hoverinfo='text', line={'color': 'yellow'},
            text='Deadhead to charger')
        fig2.add_trace(new_trace)

        # Second DH trip: charger to next trip
        if next_start_coords:
            try:
                directions2 = od_directions[dest1, next_start_coords]
            except KeyError:
                directions2 = get_gmap_directions(dest1, next_start_coords)
                od_directions[dest1, next_start_coords] = directions2

            plot_line = polyline.decode(directions2)
            lat = [pt[0] for pt in plot_line]
            lon = [pt[1] for pt in plot_line]
            new_trace = go.Scattermapbox(
                mode='lines', lat=lat, lon=lon, showlegend=False,
                hoverinfo='text', line={'color': 'yellow'},
                text='Deadhead to next trip')
            fig2.add_trace(new_trace)

    new_trace = go.Scattermapbox(
        lat=[end_coords[0], start_coords[0]],
        lon=[end_coords[1], start_coords[1]],
        hoverinfo='text', text=['Current Trip End', 'Current Trip Start'],
        mode='markers', showlegend=False,
        marker=dict(size=10, color=['red', 'green']))
    fig1.add_trace(new_trace)

    if end_coords == next_start_coords:
        label = 'Current Trip Start And Next Trip End (No Deadhead)'
        new_trace = go.Scattermapbox(
            lat=[end_coords[0]], lon=[end_coords[1]], hoverinfo='text',
            text=label, mode='markers', showlegend=False,
            marker=dict(size=20, opacity=0.4, color=['green', 'red']))
        fig2.add_trace(new_trace)

    else:
        try:
            new_trace = go.Scattermapbox(
                lat=[end_coords[0], next_start_coords[0]],
                lon=[end_coords[1], next_start_coords[1]],
                hoverinfo='text', text=['Current Trip End', 'Next Trip Start'],
                mode='markers', showlegend=False,
                marker=dict(size=20, opacity=0.4, color=['green', 'red']))
            fig2.add_trace(new_trace)
        except TypeError:
            # TODO: better handling of last trip
            pass

    fig1.update_layout(mapbox_style="dark")
    fig2.update_layout(mapbox_style="dark")
    return fig1, fig2


def plot_deadhead(result_df: pd.DataFrame, loc_df: pd.DataFrame,
                  coords_df: pd.DataFrame):
    """
    Plot deadhead trips to chargers on a map using plotly.

    :param result_df: DF of model results
    :param loc_df: DF providing charger details/locations
    :param coords_df: DF giving coordinates of bus terminals
    :return: plot of all deadhead trips
    """
    result_df = result_df.copy()
    loc_df = loc_df.copy()
    coords_df = coords_df.copy()

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

    directions_path_str = str(
        Path(__file__).resolve().parent / 'data' / 'gmaps'
        / 'deadhead_directions.pickle')

    try:
        with open(directions_path_str, 'rb') as handle:
            od_directions = pickle.load(handle)
    except FileNotFoundError:
        # If file doesn't exist, create new dict
        od_directions = dict()

    if 'trip_id' in coords_df.columns:
        coords_df.set_index('trip_id', inplace=True)

    if 'name' in loc_df.columns:
        loc_df.set_index('name', inplace=True)
    loc_df['symbol'] = 'fuel'

    # TODO: merge instead of constantly using apply()
    # Filter out dummy trips to/from depot (trip IDs 0 and 100)
    result_df = result_df[~result_df['trip_id'].isin([0, 100])]
    result_df['term_y'] = result_df.apply(
        lambda x: coords_df.at[x['trip_id'], 'end_lat'], axis=1)
    result_df['term_x'] = result_df.apply(
        lambda x: coords_df.at[x['trip_id'], 'end_lon'], axis=1)
    chg_cts = pd.DataFrame(result_df.groupby(
        ['term_x', 'term_y', 'chg_site']).count()['trip_id']).reset_index()
    chg_cts = chg_cts.rename({'trip_id': 'n_trips'}, axis=1)
    chg_cts['chg_y'] = chg_cts.apply(
        lambda x: loc_df.at[x['chg_site'], 'lat'], axis=1)
    chg_cts['chg_x'] = chg_cts.apply(
        lambda x: loc_df.at[x['chg_site'], 'lon'], axis=1)
    used_stations = list(chg_cts['chg_site'].unique())
    used_loc_df = loc_df.filter(items=used_stations, axis=0)

    fig = px.scatter_mapbox(
        used_loc_df, lat="lat", lon="lon", text='label_name',
        zoom=9, size_max=10, width=600, height=750
    )
    fig.update_traces(marker={'size': 10, 'symbol': loc_df['symbol']})
    fig.update_traces(textposition='bottom center', textfont_size=15,
                      textfont_color='black')

    # Trip terminals
    # Marker size: scale linearly from minimum to maximum
    min_marker = 10
    max_marker = 20
    msize = np.round(min_marker + (max_marker - min_marker) * (
        chg_cts['n_trips'] - chg_cts['n_trips'].min()) / chg_cts['n_trips'].max())

    trip_text = ['trips' if chg_cts.at[i, 'n_trips'] > 1
                 else 'trip' for i in chg_cts.index]
    text = ['{:d} deadhead {} from here<br>to {}'.format(
        chg_cts.at[idx, 'n_trips'], trip_text[ct], chg_cts.at[idx, 'chg_site'])
            for ct, idx in enumerate(chg_cts.index)]
    new_trace = go.Scattermapbox(
        lat=chg_cts['term_y'], lon=chg_cts['term_x'], showlegend=True,
        mode='markers',
        marker=dict(size=msize, color='rgba(60, 120, 255, 0.95)'),
        name='Trip Terminal ', text=text, hoverinfo='text')

    fig.add_trace(new_trace)

    for idx, r in chg_cts.iterrows():
        orig = chg_cts.iloc[idx][['term_y', 'term_x']].tolist()
        dest = chg_cts.iloc[idx][['chg_y', 'chg_x']].tolist()

        try:
            directions = od_directions[orig[0], dest[0]]

        except KeyError:
            directions = get_gmap_directions(orig, dest)
            od_directions[orig[0], dest[0]] = directions

        plot_line = polyline.decode(directions)
        lat = [pt[0] for pt in plot_line]
        lon = [pt[1] for pt in plot_line]
        trip_text = 'trips' if r['n_trips'] > 1 else 'trip'
        text = '{:d} {} to {}'.format(r['n_trips'], trip_text, r['chg_site'])
        new_trace = go.Scattermapbox(
            mode='lines', lat=lat, lon=lon, showlegend=False, text=text,
            hoverinfo='text', line={'color': 'red', 'width': 2}
        )
        fig.add_trace(new_trace)
        # TODO: maybe also plot deadhead to next trip

    try:
        # Trace for legend
        new_trace = go.Scattermapbox(
            mode='lines', lat=[lat[1]], lon=[lon[1]], showlegend=True,
            name='Deadhead Trip  ', line={'color': 'red', 'width': 2})
        fig.add_trace(new_trace)
    except NameError:
        pass

    # Reverse order to put markers on top
    fdata = fig.data
    fig.data = tuple(list(fdata[1:]) + [fdata[0]])
    fig = fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=0.98, font={'size': 14}),
        margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(mapbox_style='light')

    return fig


def plot_chargers(
        result_df: pd.DataFrame, zero_time: datetime.datetime,
        site: str = 'All'):
    plt.rcParams.update({'font.size': 16})
    # Get start and end times for plotting
    result_df['chg_start'] = result_df['end_time'] + result_df['dh1']
    result_df['chg_end'] = result_df['chg_start'] + result_df['chg_time']
    x_lb = result_df['start_time'].min()
    x_ub = max(result_df['end_time'].max(), result_df['chg_end'].max())
    used_sites = sorted(list(
        pd.unique(result_df[~result_df['chg_site'].isna()]['chg_site'])))
    n_sites = len(used_sites)

    # Create charging intervals
    chg_intervals = dict()
    for s in used_sites:
        s_df = result_df[result_df['chg_site'] == s]
        chg_starts = s_df['chg_start'].tolist()
        chg_ends = s_df['chg_end'].tolist()
        chg_intervals[s] = [
            (chg_starts[i], chg_ends[i]) for i in range(len(chg_starts))]

    if site == 'All':
        n_cols = 1 if n_sites <= 4 else 2
        n_rows = int(np.ceil(n_sites/n_cols))

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True,
                                sharey=True, figsize=(8*n_cols, 3*n_rows))

        for i, s in enumerate(used_sites):
            start_times = sorted([j[0] for j in chg_intervals[s]])
            end_times = sorted([j[1] for j in chg_intervals[s]])
            x = [x_lb] + sorted(start_times + end_times) + [x_ub]
            x_dt = [zero_time + datetime.timedelta(minutes=m) for m in x]
            y = [0]*len(x)
            start_idx = 0
            end_idx = 0
            while start_idx < len(start_times) and end_idx < len(end_times):
                y_idx = start_idx + end_idx + 1
                start_val = start_times[start_idx]
                end_val = end_times[end_idx]
                if start_val < end_val:
                    y[y_idx] = y[y_idx-1] + 1
                    start_idx += 1
                elif end_val < start_val:
                    y[y_idx] = y[y_idx-1] - 1
                    end_idx += 1
                else:
                    # New charge begins as soon as another ends
                    y[y_idx] = y[y_idx-1]
                    y[y_idx+1] = y[y_idx]
                    start_idx += 1
                    end_idx += 1

            if n_cols == 1:
                ax_i = axs[i]
                row_ix = i
                col_ix = 0
            else:
                row_ix = int(np.floor(i/n_cols))
                col_ix = i % n_cols
                ax_i = axs[row_ix][col_ix]

            ax_i.tick_params(
                left=True, bottom=True, labelbottom=True, labelleft=True)

            # Set ticks
            # if col_ix == 0:
            #     ax_i.tick_params(left=True)
            #
            # if row_ix == n_rows-1:
            #     ax_i.tick_params(bottom=True)

            # Add labels and plot
            ax_i.set_title('Charging Site: {}'.format(s))
            ax_i.step(x_dt, y, where='post')
            ax_i.xaxis.set_major_locator(HourLocator(interval=4))
            ax_i.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax_i.yaxis.set_major_locator(ticker.MultipleLocator())

        # Don't show axes for plots that don't exist
        if (n_cols > 1) and (n_sites % 2 != 0):
            fig.delaxes(axs[int(n_sites / 2), 1])
            ax_i = axs[int(n_sites/2)-1, 1]
            ax_i.tick_params(bottom=True)
            ax_i.xaxis.set_major_locator(HourLocator(interval=4))
            ax_i.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax_i.xaxis.set_tick_params(labelbottom=True)
            ax_i.yaxis.set_major_locator(ticker.MultipleLocator())

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                        right=False)
        plt.ylabel('Number of Buses Charging')
        plt.xlabel('Time')
        plt.tight_layout(pad=1.02)
        plt.savefig('metro_util.pdf', dpi=350, bbox_inches='tight')

    else:
        if site not in used_sites:
            raise ValueError(
                'Invalid charging site: {}. Valid options are: {}'.format(
                    site, used_sites))

        fig, ax = plt.subplots(figsize=(8, 3))
        start_times = sorted([j[0] for j in chg_intervals[site]])
        end_times = sorted([j[1] for j in chg_intervals[site]])
        x = [x_lb] + sorted(start_times + end_times) + [x_ub]
        x_dt = [zero_time + datetime.timedelta(minutes=m) for m in x]
        y = [0] * len(x)
        start_idx = 0
        end_idx = 0
        while start_idx < len(start_times) and end_idx < len(end_times):
            y_idx = start_idx + end_idx + 1
            start_val = start_times[start_idx]
            end_val = end_times[end_idx]
            if start_val < end_val:
                y[y_idx] = y[y_idx - 1] + 1
                start_idx += 1
            elif end_val < start_val:
                y[y_idx] = y[y_idx - 1] - 1
                end_idx += 1
            else:
                # New charge begins as soon as another ends
                y[y_idx] = y[y_idx - 1]
                y[y_idx + 1] = y[y_idx]
                start_idx += 1
                end_idx += 1

        ax.set_title('Charging Site: {}'.format(site))
        ax.step(x_dt, y, where='post')
        ax.xaxis.set_major_locator(HourLocator(interval=4))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.yaxis.set_major_locator(ticker.MultipleLocator())
        plt.ylabel('Buses Charging')
        plt.xlabel('Time')
        plt.tight_layout(pad=1.02)
        plt.savefig('metro_util.pdf', dpi=350, bbox_inches='tight')

    return fig


def plot_charging_and_dh_comparison(result_df: pd.DataFrame):
    """
    Create a scatter plot that shows charging behavior, including charge
    durations and deadheading time.
    :param result_df: DataFrame created by reading in CSV results of model
    :return: plot figure
    """
    chg_df = result_df[result_df['chg_site'].notna()]
    # fig, ax = plt.subplots(figsize=(16, 9))
    chg_df['net_dh'] = chg_df['dh1'] + chg_df['dh2'] - chg_df['dh3']
    # ax.scatter(chg_df['net_dh'], chg_df['chg_time'])
    fig = px.scatter(
        chg_df, x='net_dh', y='chg_time', color='chg_site',
        title='Charging Versus Deadhead Time',
        labels={
            'net_dh': 'Net Deadhead Time (min.)',
            'chg_time': 'Charging Duration (min.)',
            'dh1': 'Deadhead to charger (min.)',
            'dh2': 'Deadhead back from charger (min.)',
            'dh3': 'Deadhead if no charging done',
            'block_id': 'Block ID',
            'trip_idx': 'Trip Number',
            'chg_site': 'Charging Site'},
        hover_data={
            'dh1': ':.2f',
            'dh2': ':.2f',
            'dh3': ':.2f',
            'chg_time': ':.2f',
            'chg_site': False,
            'block_id': True,
            'trip_idx': True
        }
    )
    # ax.set_xlabel('Net Deadhead Time (minutes)')
    # ax.set_ylabel('Charging Time (minutes)')
    # ax.set_title('Charging Time Versus Deadhead Time')

    # Plot y=x line
    xmax = chg_df['net_dh'].max()
    fig.add_trace(
        go.Scatter(x=[0, xmax], y=[0, xmax], mode='lines', name='y=x'))
    fig.update_layout(
        title='Charging Versus Deadhead Time')
    return fig


def plot_charger_timelines_old(fname, zero_time=None):
    """
    Generate interactive timeline plots showing all activity at each
    charger using plotly. One plot will be generated for each charger
    that is used at least once.

    :param fname: filename of CSV giving results from charge scheduling
        model
    :param zero_time: datetime.datetime giving reference time
        corresponding to zero in the model, defaults to midnight 1/1/24
    """
    # TODO: colors are off for some reason, looks way worse than it did
    #   in Jupyter
    if zero_time is None:
        zero_time = datetime.datetime(2024, 1, 1, 0, 0)

    df = pd.read_csv(fname)
    df['block_id'] = df['block_id'].astype(str)
    df = df[df['chg_time'] > 0]

    for c in df['charger'].unique():
        c_df = df[df['charger'] == c].copy()
        c_df['plugin_time'] = zero_time + pd.to_timedelta(c_df['plugin_time'],
                                                          unit='minute')
        c_df['finish_chg_time'] = c_df['plugin_time'] + pd.to_timedelta(
            c_df['chg_time'], unit='minute')
        # Sort blocks by first plugin time
        block_order = c_df.sort_values(by='plugin_time', ascending=True)[
            'block_id'].unique().tolist()
        fig = px.timeline(
            c_df, x_start='plugin_time', x_end='finish_chg_time', y='block_id',
            category_orders={'block_id': block_order},
            title='Charger Utilization at {}'.format(c),
            labels={
                'plugin_time': 'Plugin Time',
                'block_id': 'Block ID of Charging Bus'
            }
        )
        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'scale': 3
            }
        }
        fig.show(config=config)


def plot_charger_timelines(df, zero_time, show_service=False, highlight=False,
                           title_fmt='{}'):
    df['block_id'] = df['block_id'].astype(str)
    # Define relevant times
    t_start = zero_time + pd.to_timedelta(
        df['start_time'].min(), unit='minute') - datetime.timedelta(hours=1)
    t_end = zero_time + pd.to_timedelta(
        df['end_time'].max(), unit='minute') + datetime.timedelta(hours=1)
    t_srs = pd.date_range(
        t_start, t_end, freq='min').to_series()
    highlight_df = t_srs.between(
        datetime.datetime(2023, 12, 6, 7, 0),
        datetime.datetime(2023, 12, 6, 9, 0)).to_frame(
        name='peak')
    trip_col_labels = {
        'start_time': 'Start Time',
        'end_time': 'End Time',
        'delay': 'Delay'
    }

    df['Delayed'] = df['delay'] > 0
    df['Delayed'] = df['Delayed'].map(
        {True: 'Delayed Trip', False: 'On-Time Trip'})
    df['trip_name'] = 'Block ' + df['block_id'].astype(str) \
                      + ', Trip ' + df['trip_idx'].astype(str)

    fig_list = list()
    for c in df['charger'].unique():
        c_df = df[df['charger'] == c].copy()
        c_df['start_time'] = zero_time + pd.to_timedelta(
            c_df['start_time'] + c_df['delay'], unit='minute')
        c_df['end_time'] = zero_time + pd.to_timedelta(
            c_df['end_time'] + c_df['delay'], unit='minute')
        c_df['plugin_time'] = zero_time + pd.to_timedelta(c_df['plugin_time'],
                                                          unit='minute')
        c_df['finish_chg_time'] = c_df['plugin_time'] + pd.to_timedelta(
            c_df['chg_time'], unit='minute')

        # Make subplots to highlight time block with delays
        fig = make_subplots(specs=[[{'secondary_y': True}]])

        # Highlight delayed period
        if highlight:
            hl_trace = go.Scatter(
                x=highlight_df.index, y=highlight_df['peak'],
                fill='tonexty', fillcolor='rgba(240, 228, 66, 0.7)',
                line_shape='hv', line_color='rgba(0,0,0,0)',
                showlegend=False
            )
            fig.add_trace(hl_trace, row=1, col=1, secondary_y=False)

        else:
            # Add a dummy trace. Skipping this makes the formatting weird because
            # there is then only one y axis used.
            fig.add_trace(
                go.Scatter(
                    x=[t_start], y=[0],
                    fillcolor='rgba(0,0,0,0)', line_color='rgba(0,0,0,0)',
                    showlegend=False, hoverinfo='skip'),
                row=1, col=1, secondary_y=False)

        # Hide highlight axis
        fig.update_xaxes(showgrid=False)
        fig.update_layout(yaxis1_range=[0, 0.5], yaxis1_showgrid=False,
                          yaxis1_showticklabels=False)

        # Only include blocks that charge here
        time_by_block = c_df.groupby('block_id')['chg_time'].sum()
        c_blocks = time_by_block[time_by_block > 0.1].index.tolist()
        c_df = c_df[c_df['block_id'].isin(c_blocks)]

        # Sort blocks by first plugin time
        order_df = c_df[c_df['chg_time'] > 0.1]
        block_order = order_df.sort_values(by='plugin_time', ascending=True)[
            'block_id'].unique().tolist()
        block_order.reverse()
        # Add trip timeline
        if show_service:
            # There is a weird bug in plotly where it only plots one color when using subplots.
            # To get around this we'll add two traces manually.
            on_time_trips = c_df[c_df['Delayed'] == 'On-Time Trip']
            delayed_trips = c_df[c_df['Delayed'] == 'Delayed Trip']

            trip_hover = {
                'Delayed': False,
                'block_id': False,
                'trip_idx': False,
                'delay': ':.2f',
                'start_time': True,
                'end_time': True
            }

            if len(on_time_trips) > 0:
                tl_ontime = px.timeline(
                    on_time_trips,
                    x_start='start_time',
                    x_end='end_time',
                    y='block_id',
                    category_orders={'block_id': block_order},
                    range_x=[t_start, t_end],
                    hover_name='trip_name',
                    hover_data=trip_hover,
                    labels=trip_col_labels,
                    color='Delayed',
                    color_discrete_map={'Delayed Trip': 'rgba(213, 94, 0, 1)',
                                        'On-Time Trip': 'rgba(128, 128, 128, 1)'}
                )
                fig.add_trace(tl_ontime.data[0], secondary_y=True)

            if len(delayed_trips) > 0:
                trip_name = 'Block ' + delayed_trips['block_id'].astype(str) \
                            + ', Trip ' + delayed_trips['trip_idx'].astype(str)
                tl_delayed = px.timeline(
                    delayed_trips,
                    x_start='start_time',
                    x_end='end_time',
                    y='block_id',
                    hover_name=trip_name,
                    hover_data=trip_hover,
                    labels=trip_col_labels,
                    category_orders={'block_id': block_order},
                    range_x=[t_start, t_end],
                    color='Delayed',
                    color_discrete_map={'Delayed Trip': 'rgba(213, 94, 0, 1)',
                                        'On-Time Trip': 'rgba(128, 128, 128, 1)'}
                )
                fig.add_trace(tl_delayed.data[0], secondary_y=True)

        # Add charger utilization timeline
        # Filter out trips without charging
        c_df = c_df[c_df['chg_time'] > 0.1]
        c_df['Status'] = 'Charging'
        tl_chg = px.timeline(
            c_df,
            x_start='plugin_time',
            x_end='finish_chg_time',
            y='block_id',
            category_orders={'block_id': block_order},
            labels={
                'plugin_time': 'Plugin Time',
                'block_id': 'Block ID',
                'finish_chg_time': 'Charging End Time'
            },
            range_x=[t_start, t_end],
            color='Status',
            color_discrete_map={'Charging': 'rgba(0,114,178, 1)'},
            hover_name='trip_name',
            hover_data={
                'Status': False,
                'block_id': False,
                #                 'trip_idx': False,
                #                 'delay': ':.2f',
                'plugin_time': True,
                'finish_chg_time': True
            }
        )
        fig.add_trace(tl_chg.data[0], secondary_y=True)

        # Clean up formatting
        # Sort y axis. This gets lost even though we specified it when making the timeline.
        fig.update_layout(
            yaxis2_categoryorder='array', yaxis2_categoryarray=block_order,
            yaxis2_side='left',
            yaxis2_tickfont_size=10, yaxis2_title='Block ID')
        fig.update_layout(
            dict(
                barmode='overlay',
                title=title_fmt.format(c),
                #                 yaxis_title='Bus',
                xaxis={'range': [t_start, t_end]},
                margin=dict(l=20, r=20, t=40, b=20)
            )
        )
        fig.update_layout(legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=0.95
        ))
        #         fig.update_yaxes(type='category')

        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'scale': 3
            }
        }
        fig.show(config=config)
        fig_list.append(fig)

    return fig_list