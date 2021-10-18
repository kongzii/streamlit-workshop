import time
import random
import numpy as np
import plotly.express as px
import pandas as pd
import streamlit as st

from geopy.geocoders import Nominatim


@st.cache
def get_best_time(location):
    return random.randint(10, 14), random.randint(0, 59)


@st.cache(allow_output_mutation=True)
def create_geolocator():
    return Nominatim(user_agent="Demo")


def primetime():
    geolocator = create_geolocator()

    c1, c2 = st.columns(2)

    with c1:
        user_location = st.text_input('Your location')

    with c2:
        n_suggests = st.number_input('How many times to suggest?', min_value=1, max_value=4, value=1)

    if user_location in (None, ''):
        st.warning('Please write your location for PrimeTime analysys.')
        st.stop()

    location = geolocator.geocode(user_location)

    if location is None:
        st.warning('Sorry, your location was not found.')
        st.stop()

    map_data = pd.DataFrame.from_dict({
        'lat': [location.latitude],
        'lon': [location.longitude],
    })

    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)

    best_hour, best_min = get_best_time(location)

    cols = st.columns(n_suggests)

    for c in cols:
        c.metric(label="Best Time", value=f'{best_hour}:{best_min}', delta="+10% reach")

    c1, c2 = st.columns(2)

    with c1:
        st.map(map_data)

    with c2:
        placeholder = st.empty()
        randomness = 1 - st.slider('Precision', min_value=0.0, max_value=1.0, value=1.0)

    with placeholder.container():
        fig = px.line(
            x=list(range(24 * 60)),
            y=-np.cos(np.linspace(0, 2 * np.pi * n_suggests, 24 * 60)) + np.random.rand(24 * 60) * randomness,
            labels={'x': 'time', 'y': 'reach)'},
        )

        st.plotly_chart(fig, use_container_width=True)
