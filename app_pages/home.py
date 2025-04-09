import streamlit as st


def app():
    st.title("Cherry Leaf Mildew Detector")
    st.markdown("""
    Welcome to the Cherry Leaf Mildew Detector Dashboard!

    **Business Need:**
    Farmy & Foods aims to automate the identification of powdery mildew in
     cherry crops to avoid low-quality harvests and reduce labor-intensive
     inspection.

    **Project Goals:**
    - Visually study healthy vs. mildew-affected leaves
    - Train a model to detect mildew from images
    - Offer an easy-to-use prediction interface
    """)
