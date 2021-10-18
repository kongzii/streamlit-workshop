import io
import requests
import streamlit as st

from webcam import webcam

MODEL_API = 'http://dl-innovations4.us-w2.aws.ccl:5000'


def cloths_recommender():
    st.markdown('''
    # Welcome to our shiny new prototype!

    Full markdown support to render our texts:

    - 1
    - 2
    - 3

    ---
    ''')

    if 'enable_camera' not in st.session_state:
        st.session_state['enable_camera'] = False

    if not st.session_state['enable_camera']:
        if st.button('Or take picture from your camera?'):
            st.session_state['enable_camera'] = True
            input_image = None

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            input_image = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'], help='Upload photo from your computer to predict something.')

        input_image_for_upload = input_image

    else:
        if st.button('Or upload image from computer?'):
            st.session_state['enable_camera'] = False
            input_image = None

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            input_image = webcam()  # Return PIL.Image

        if input_image is not None:
            input_image_for_upload = io.BytesIO()
            input_image.save(input_image_for_upload, 'png')
            input_image_for_upload.seek(0)

    if input_image is None:
        st.warning('Please input some image.')
        st.stop()

    with input_col2:
        st.image(input_image)

    with st.spinner('Analyzing...'):
        try:
            resp = requests.post(
                f'{MODEL_API}/search/',
                files=dict(upload_file=input_image_for_upload),
                timeout=5,
            ).json()

        except Exception as e:
            resp = None

            st.error(f'Sorry, can not connect to the model. Please contant innovations with following message: {e}.')
            st.stop()

    if resp['error']:
        st.error(f'Sorry. {resp["error"]}')
        st.stop()

    st.header("You look like")

    for i, col in enumerate(st.columns(len(resp['results']['similar_person_images']))):
        with col:
            st.image(resp['results']['similar_person_images'][i])

    st.header("We recommend you")
    st.text(' and '.join(f'{a} {b}' for a, b in resp['results']['recomm_clothes_pairs']))

    if resp['results']['recomm_image_paths']:
        for path in resp['results']['recomm_image_paths']:
            st.image(f"{MODEL_API}/{path}")

    else:
        st.text('Sorry, no exampels of these clothes.')
