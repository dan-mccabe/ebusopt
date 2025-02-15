import streamlit as st
from pathlib import Path
from ebusopt.zebra.utils import config_page

st.set_page_config(
    page_title='ZEBRA Home',
    page_icon='🦓',
    layout='wide'
)

config_page()

for k, v in st.session_state.items():
    st.session_state[k] = v


def main():
    st.image(str(Path(__file__).resolve().parent / 'iuts_header.png'))
    about_file = Path(__file__).resolve().parent / 'project_overview.md'
    about_text = about_file.read_text()
    st.markdown(about_text)

    # Initialize page visit booleans
    for i in ['gtfs_run', 'range_set']:
        if i not in st.session_state:
            st.session_state[i] = False


if __name__ == '__main__':
    main()
