import streamlit as st


def app():
    st.title("Model Performance")

    st.markdown("### Accuracy Over Epochs")
    st.image(
        "outputs/03_modelling_and_evaluating/training_accuracy.png",
        caption="Model Training Accuracy"
    )

    st.markdown("### Loss Over Epochs")
    st.image(
        "outputs/03_modelling_and_evaluating/training_loss.png",
        caption="Model Training Loss"
    )

    st.markdown("""
    ### Conclusion
    The model achieved **over 98% accuracy** on test data, demonstrating high
     predictive power and strong generalization to unseen leaf images.
    """)
