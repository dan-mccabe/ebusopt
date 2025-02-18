import streamlit as st
import requests
import zipfile
import io
import os
import shutil
import time
import logging
import datetime
import pandas as pd
from pathlib import Path
from ebusopt.gtfs_beb import GTFSData, GTFSError
from ebusopt.zebra.utils import config_page, read_markdown

st.set_page_config(
    page_title='Specify GTFS Data',
    page_icon='🦓',
    layout='wide'
)
config_page()


def reset_session_state():
    # Delete function cache
    get_day_trips.clear()
    for k in list(st.session_state.keys()):
        # Keep the gtfs_choice dropdown selection, as it impacts what
        # we do in reset_gtfs
        if k != 'gtfs_choice':
            del st.session_state[k]
    for k in ['gtfs_run', 'show_date_input', 'range_set']:
        st.session_state[k] = False


# Get trips on a single day
@st.cache_data(show_spinner=False)
def get_day_trips(_gtfs, input_date):
    return _gtfs.get_trips_from_date(input_date)


@st.cache_data(show_spinner=False)
def load_gtfs(agency):
    """Load GTFS data and let Streamlit handle caching."""
    if agency == 'metro':
        dir_name = 'metro_may24'
    elif agency == 'trimet':
        dir_name = 'trimet_may24'
    else:
        raise ValueError(f'Unknown agency {agency}')

    path_here = Path(__file__).absolute()
    gtfs_path = path_here.parent.parent.parent / 'data' / 'gtfs' / dir_name
    gtfs = GTFSData.from_dir(dir_name=str(gtfs_path))
    # Load pre-computed shapes summary
    try:
        gtfs.shapes_summary_df = pd.read_csv(gtfs_path / 'shapes_summary.csv')
    except FileNotFoundError:
        # If the file is missing, that's fine. We'll calculate manually
        print('couldn\'t find shapes_summary.csv')
        pass

    return gtfs


# Callback to control display of detailed inputs
def reset_gtfs():
    reset_session_state()
    st.session_state.gtfs_changed = True


# UI code starts here
st.markdown(
    "<h2 style='text-align: center; color: black;'>Specify GTFS Files</h1>",
    unsafe_allow_html=True)

t1, t2 = st.tabs(['📄 Select Data', '❓ Help'])
if 'show_date_input' not in st.session_state:
    st.session_state.show_date_input = False

with t2:
    st.markdown(read_markdown('ebusopt/zebra/text/gtfs_data_help.md'))

with t1:
    choice_col, _ = st.columns([0.4, 0.6])
    if 'gtfs_choice' not in st.session_state:
        st.session_state['gtfs_choice'] = 'Please select an option'
    with choice_col:
        st.selectbox(
            label='Choose how to specify GTFS data',
            options=[
                'Please select an option',
                'Use King County Metro data',
                'Use TriMet data',
                'Upload my own files'],
            on_change=reset_gtfs,
            key='gtfs_choice'
        )

    if st.session_state['gtfs_choice'] == 'Please select an option':
        st.stop()

    elif st.session_state.gtfs_choice == 'Upload my own files':
        gtfs_11, gtfs_12 = st.columns(2)
        cal_file = gtfs_11.file_uploader('calendar.txt')
        cal_dates_file = gtfs_12.file_uploader('calendar_dates.txt')
        gtfs_21, gtfs_22 = st.columns(2)
        trips_file = gtfs_21.file_uploader('trips.txt')
        shapes_file = gtfs_22.file_uploader('shapes.txt')
        gtfs_31, gtfs_32 = st.columns(2)
        stop_times_file = gtfs_31.file_uploader('stop_times.txt')
        routes_file = gtfs_32.file_uploader('routes.txt')

        if st.button('Upload all files'):
            missing_files = [
                f for f in [
                    shapes_file, trips_file, routes_file, stop_times_file]
                if f is None
            ]
            if missing_files:
                st.error(
                    'One or more required files are missing. Please include '
                    'all required files to proceed. See **❓ Help** for '
                    'details.'
                )
                st.stop()

            wait_msg =\
                'Please wait while your files are processed. You will see a ' \
                'green success message when it is time to proceed.'
            with st.spinner(wait_msg):
                reset_session_state()
                st.session_state.gtfs_changed = True
                if cal_file is None:
                    cal_file = ''

                if cal_dates_file is None:
                    cal_dates_file = ''

                try:
                    st.session_state.gtfs = GTFSData(
                        calendar_file=cal_file,
                        calendar_dates_file=cal_dates_file,
                        shapes_file=shapes_file,
                        trips_file=trips_file,
                        routes_file=routes_file,
                        stop_times_file=stop_times_file
                    )

                except GTFSError as err:
                    st.error(
                        'There was an error with one or more of the files you '
                        'uploaded. Most likely, a column required by ZEBRA '
                        'is missing in your tables. Please see the error '
                        'trace below and the **❓ Help** tab for more '
                        'information.'
                    )
                    st.write(err)
                    st.markdown('#')
                    st.stop()
            st.success('Successfully input GTFS data.')
            st.session_state.show_date_input = True

    elif st.session_state.gtfs_choice == 'Specify with a link':
        # This method of specifying data is currently deprecated.
        st.markdown(
            'In the text box below, enter the complete URL to a .zip file of '
            'the GTFS feed you wish to analyze. The default value points to King '
            'Count Metro\'s current GTFS files.')

        if 'gtfs_url' not in st.session_state:
            st.session_state.gtfs_url = \
                'https://metro.kingcounty.gov/GTFS/google_transit.zip'
        st.text_input(
            'Link to GTFS feed', key='gtfs_url'
        )

        st.markdown(
            'Click the button below to download GTFS files. This may take a '
            'couple of minutes.'
        )
        if st.button('Download GTFS files'):
            request_start = time.time()
            r = requests.get(st.session_state.gtfs_url)
            zip_start = time.time()
            logging.debug(
                'Time to get files: {:.1f} s'.format(zip_start - request_start))
            z = zipfile.ZipFile(io.BytesIO(r.content))

            # Make the directory if it doesn't exist already
            try:
                shutil.rmtree('st_gtfs_tmp')
            except FileNotFoundError:
                pass
            os.mkdir('st_gtfs_tmp')

            z.extractall(path='st_gtfs_tmp')
            logging.debug(
                'Time to unzip files: {:.1f} s'.format(time.time() - zip_start))
            st.session_state.gtfs = GTFSData.from_dir('st_gtfs_tmp')
            shutil.rmtree('st_gtfs_tmp')
            st.session_state.show_date_input = True

    elif st.session_state.gtfs_choice == 'Use King County Metro data':
        with st.spinner('Loading GTFS files. Please wait.'):
            st.session_state.gtfs = load_gtfs('metro')
        st.session_state.show_date_input = True

    elif st.session_state.gtfs_choice == 'Use TriMet data':
        with st.spinner('Loading GTFS files. Please wait.'):
            st.session_state.gtfs = load_gtfs('trimet')
        st.session_state.show_date_input = True

    else:
        raise ValueError('Invalid choice of GTFS source: {}'.format(
            st.session_state.gtfs_choice))

    if not st.session_state.show_date_input:
        st.stop()

    st.markdown('----')
    st.markdown('### 🗓️ Select a Date for Analysis')

    # Identify busiest day
    day_totals = st.session_state.gtfs.get_n_trips_per_day()
    max_days = day_totals[
        day_totals['n_trips'] == day_totals['n_trips'].max()]
    first_max_day = max_days.sort_index().index[0]

    min_date = min(day_totals.index)
    max_date = max(day_totals.index)

    dates_str = (
        'The provided GTFS files cover a date range from **:green[{}]** to '
        '**:red[{}]**. Please select a single day to serve as the base of our '
        'analysis. The default value below corresponds to the day with the '
        'greatest number of active trips: **:blue[{}]**.'.format(
            min_date.strftime('%m/%d/%Y'), max_date.strftime('%m/%d/%Y'),
            first_max_day.strftime('%m/%d/%Y'))
    )

    st.markdown(dates_str)
    c1, _ = st.columns([0.3, 0.7])

    if 'date_set' not in st.session_state:
        st.session_state.date_set = first_max_day
    c1.date_input(
        '**Select a Date**', value=None, min_value=min_date,
        max_value=max_date, key='date_set')

    st.session_state.day_trips = get_day_trips(
        st.session_state.gtfs, datetime.datetime.combine(
            st.session_state.date_set, datetime.time(0)))

    st.markdown(
        'There are **:blue[{} trips]** and **:blue[{} blocks]** serving '
        '**:blue[{} routes]** active on this day.'.format(
            len(st.session_state.day_trips),
            len(st.session_state.day_trips['block_id'].unique()),
            len(st.session_state.day_trips['route_id'].unique())
        ))

    st.session_state.gtfs_run = True

    st.markdown('#')

