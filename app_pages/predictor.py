import streamlit as st
import pandas as pd
from PIL import Image

from src.data_management import load_model_and_metadata
from src.machine_learning.predictive_analysis import (
    preprocess_uploaded_image,
    predict_from_array,
)

# ===== Paths (relative to repo root) =====
MODEL_PATH = "outputs/v1/cherry_leaf_mildew_model.h5"  # loader will also try .keras
IMAGE_SHAPE_PATH = "outputs/02_data_visualisation/image_shape.pkl"

# IMPORTANT: Pick the path where your file actually is.
# If you trained with the notebooks, it’s usually under jupyter_notebooks/...
CLASS_INDEX_PATH = "outputs/03_modelling_and_evaluating/class_indices.pkl"
# If you manually moved it to outputs/, then use this instead:
# CLASS_INDEX_PATH = "outputs/03_modelling_and_evaluating/class_indices.pkl"


@st.cache_resource(show_spinner=False)
def get_artifacts():
    """
    Load model + metadata once, cache across Streamlit reruns.
    Will auto-download the model if HF_REPO_ID/HF_FILENAME or MODEL_URL are set.
    """
    try:
        model, image_shape, class_indices = load_model_and_metadata(
            MODEL_PATH, IMAGE_SHAPE_PATH, CLASS_INDEX_PATH
        )
        return model, image_shape, class_indices
    except FileNotFoundError as e:
        # Give a friendly message about how to provide the model
        st.error(
            f"Model/metadata missing: {e}\n\n"
            "Fix one of the following:\n"
            "1) Place your model locally at 'outputs/v1/cherry_leaf_mildew_model.h5' "
            "(or .keras), and ensure the two .pkl paths exist.\n"
            "2) Or set environment variables so the app can download it:\n"
            "   - HF_REPO_ID and HF_FILENAME (Hugging Face Hub), "
            "and optionally HUGGINGFACE_HUB_TOKEN for private repos; or\n"
            "   - MODEL_URL for a direct HTTP(S) link.\n"
        )
        raise
    except Exception as e:
        st.error(f"Unexpected error while loading artifacts: {e}")
        raise


def app():
    st.title("Predict Cherry Leaf Condition")

    # Load artifacts once (cached)
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

        # ---- Responsive grid: up to 3 columns per row ----
        n_cols = min(3, len(uploaded_files))
        cols = st.columns(n_cols)

        for i, file in enumerate(uploaded_files):
            # (Re)create row of columns every n_cols items
            if i % n_cols == 0 and i != 0:
                cols = st.columns(n_cols)
            col = cols[i % n_cols]

            with col:
                st.image(file, caption=file.name, use_container_width=True)  # scales nicely
                img_array = preprocess_uploaded_image(file, image_shape)
                pred_class, confidence = predict_from_array(model, img_array, class_indices)

                st.markdown(f"**Prediction:** {pred_class}")
                st.markdown(f"**Confidence:** {confidence:.2%}")

                results.append(
                    {"Image": file.name, "Prediction": pred_class, "Confidence": f"{confidence:.2%}"}
                )

        # === Results Table ===
        st.markdown("---")
        st.markdown("### 📊 Prediction Summary")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        # === Download Button ===
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=df_results.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )
