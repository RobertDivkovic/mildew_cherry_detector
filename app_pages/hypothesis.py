import streamlit as st


def app():
    st.title("Hypothesis & Validation")
    st.markdown("""
    ### Hypothesis
    Powdery mildew leaves have distinguishable visual features that can be
     identified using computer vision techniques.

    ### Validation
    - Visual difference study confirms clear image-level patterns
    - Model performance (98%+ accuracy) supports strong predictive capability
    """)
