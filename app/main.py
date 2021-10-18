import streamlit as st

import other
import primetime
import cloths_recommender


# Config

st.set_page_config(
    page_title='Streamlit Workshop',
    layout='wide',
    menu_items={
        'Get help': 'https://techbakers.slack.com/archives/C0BUNNK70',
        'Report a Bug': None,
        'About': 'Streamlit Workshop in Live!',
    },
    initial_sidebar_state='collapsed',
)

# Page Select

selected_demo = st.sidebar.selectbox(
    'Demo',
    options=[
        'Primetime',
        'Cloths Recommender',
        'Other',
    ],
)

# Application

if selected_demo == 'Cloths Recommender':
    cloths_recommender.cloths_recommender()

elif selected_demo == 'Primetime':
    primetime.primetime()

elif selected_demo == 'Other':
    other.other()

else:
    raise RuntimeError('Invalid option??')
