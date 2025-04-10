import streamlit as st
import pandas as pd
from PIL import Image

from src.data_management import load_model_and_metadata
from src.machine_learning.predictive_analysis import preprocess_uploaded_image, predict_from_array

# === Load model and metadata ===
MODEL_PATH = "outputs/v1/cherry_leaf_mildew_model.h5"
IMAGE_SHAPE_PATH = "outputs/02_data_visualisation/image_shape.pkl"
CLASS_INDEX_PATH = "outputs/03_modelling_and_evaluating/class_indices.pkl"

model, image_shape, class_indices = load_model_and_metadata(
    MODEL_PATH, IMAGE_SHAPE_PATH, CLASS_INDEX_PATH
)


# === App Function ===
def app():
    st.title("Predict Cherry Leaf Condition")

    uploaded_files = st.file_uploader(
        "Upload cherry leaf image(s):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("### 🖼 Uploaded Images")
        results = []
        cols = st.columns(len(uploaded_files))

        for i, file in enumerate(uploaded_files):
            with cols[i]:
                st.image(file, caption=file.name, width=200)

                img_array = preprocess_uploaded_image(file, image_shape)
                pred_class, confidence = predict_from_array(model, img_array, class_indices)

                st.markdown(f"**Prediction:** {pred_class}")
                st.markdown(f"**Confidence:** {confidence:.2%}")

                results.append({
                    "Image": file.name,
                    "Prediction": pred_class,
                    "Confidence": f"{confidence:.2%}"
                })

        # === Results Table ===
        st.markdown("---")
        st.markdown("### 📊 Prediction Summary")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)

        # === Download Button ===
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=df_results.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
