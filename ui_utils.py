import os
import base64
import streamlit as st

def set_background(image_file):
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, image_file)
    with open(full_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
