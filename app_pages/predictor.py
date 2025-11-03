# app_pages/predictor.py
import streamlit as st
import pandas as pd
from pathlib import Path

from src.data_management import load_model_and_metadata
from src.machine_learning.predictive_analysis import (
    preprocess_uploaded_image,
    predict_from_array,
)

# This file path is: <repo>\app_pages\predictor.py
# => parents[1] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = REPO_ROOT / "outputs" / "v1" / "cherry_leaf_mildew_model.h5"
IMAGE_SHAPE_PATH = REPO_ROOT / "outputs" / "02_data_visualisation" / "image_shape.pkl"
CLASS_INDEX_PATH = REPO_ROOT / "outputs" / "03_modelling_and_evaluating" / "class_indices.pkl"

@st.cache_resource(show_spinner=False)
def get_artifacts():
    try:
        return load_model_and_metadata(
            str(MODEL_PATH), str(IMAGE_SHAPE_PATH), str(CLASS_INDEX_PATH)
        )
    except FileNotFoundError as e:
        st.error(
            f"Model/metadata missing: {e}\n\n"
            f"Checked paths:\n"
            f"- model: `{MODEL_PATH}`\n"
            f"- image_shape: `{IMAGE_SHAPE_PATH}`\n"
            f"- class_indices: `{CLASS_INDEX_PATH}`\n"
        )
        raise


def app():
    st.title("Predict Cherry Leaf Condition")

    with st.spinner("Loading model and metadata..."):
        model, image_shape, class_indices = get_artifacts()

    uploaded_files = st.file_uploader(
        "Upload cherry leaf image(s):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown("### 🖼 Uploaded Images")
        results = []

        # Responsive grid (max 3 columns)
        n_cols = min(3, len(uploaded_files))
        cols = st.columns(n_cols)

        for i, file in enumerate(uploaded_files):
            if i % n_cols == 0 and i != 0:
                cols = st.columns(n_cols)
            col = cols[i % n_cols]

            with col:
                st.image(file, caption=file.name, use_container_width=True)
                img_array = preprocess_uploaded_image(file, image_shape)
                pred_class, confidence = predict_from_array(model, img_array, class_indices)

                st.markdown(f"**Prediction:** {pred_class}")
                st.markdown(f"**Confidence:** {confidence:.2%}")

                results.append(
                    {"Image": file.name, "Prediction": pred_class, "Confidence": f"{confidence:.2%}"}
                )

        st.markdown("---")
        st.markdown("### 📊 Prediction Summary")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=df_results.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )
