import streamlit as st
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import numpy as np
import random
import itertools
import io


def app():
    st.title("Visual Differentiation Study")
    st.markdown("""
    This section highlights the visual characteristics of healthy and mildew-infected cherry leaves using image processing techniques.
    """)

    # Average & variability images
    if st.checkbox("Show Average and Variability Images"):
        st.image(
            "outputs/02_data_visualisation/avg_var_healthy.png",
            caption="Healthy Leaf: Average & Variability", width=500)
        st.image(
            "outputs/02_data_visualisation/avg_var_powdery_mildew.png",
            caption="Powdery Mildew Leaf: Average & Variability", width=500)
        st.info("There is a clear visual difference in texture and brightness between healthy and infected leaves.")

    # Difference between average images
    if st.checkbox("Show Difference Between Average Images"):
        st.image(
            "outputs/02_data_visualisation/avg_diff.png",
            caption="Difference Between Average Images", width=1000)
        st.warning("Although subtle, the darker and green-centered regions help differentiate mildew infection.")

    # Image Montage Section
    if st.checkbox("Generate Image Montage"):
        label_options = ["healthy", "powdery_mildew"]
        selected_label = st.selectbox("Select Leaf Condition", label_options)
        nrows = st.slider("Number of rows", 1, 6, 4)
        ncols = st.slider("Number of columns", 1, 6, 3)
        figsize_val = st.slider("Figure size (width, height)", 5, 20, (12, 10))

        if st.button("Create Montage"):
            montage_buffer = image_montage(
                dir_path="inputs/cherry_leaves_split/train",
                label=selected_label,
                nrows=nrows, ncols=ncols,
                figsize=figsize_val
            )
            if montage_buffer:
                st.download_button(
                    label="Download Montage as PNG",
                    data=montage_buffer,
                    file_name="montage.png",
                    mime="image/png"
                )

    # Image dimension distribution
    if st.checkbox("Show Image Dimension Distribution"):
        label_dirs = ["healthy", "powdery_mildew"]
        all_widths, all_heights = [], []
        for lbl in label_dirs:
            path = os.path.join("inputs/cherry_leaves_split/train", lbl)
            for file in os.listdir(path):
                img = imread(os.path.join(path, file))
                all_heights.append(img.shape[0])
                all_widths.append(img.shape[1])

        fig, ax = plt.subplots(figsize=(2, 1))
        sns.scatterplot(x=all_widths, y=all_heights, alpha=0.5, ax=ax)
        ax.set_title("Image Dimension Distribution")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        st.pyplot(fig)
        st.info(f"Total images analyzed: {len(all_widths)}")

    # Image intensity histogram
    if st.checkbox("Show Image Intensity Histogram"):
        label = st.selectbox("Select Label for Histogram", ["healthy", "powdery_mildew"], key="hist")
        path = os.path.join("inputs/cherry_leaves_split/train", label)
        pixel_values = []
        for file in os.listdir(path)[:50]:
            img = imread(os.path.join(path, file))
            gray = np.mean(img, axis=2)
            pixel_values.extend(gray.flatten())

        fig, ax = plt.subplots()
        sns.histplot(pixel_values, bins=50, kde=True, ax=ax, color="gray")
        ax.set_title(f"Pixel Intensity Distribution - {label.capitalize()}")
        ax.set_xlabel("Pixel Intensity")
        st.pyplot(fig)


def image_montage(dir_path, label, nrows, ncols, figsize=(12, 10)):
    sns_dir = os.path.join(dir_path, label)
    if not os.path.exists(sns_dir):
        st.error("Directory not found for the selected label.")
        return None

    images = os.listdir(sns_dir)
    if nrows * ncols > len(images):
        st.warning(f"Not enough images in the folder. Found {len(images)}.")
        return None

    selected_images = random.sample(images, nrows * ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plot_positions = list(itertools.product(range(nrows), range(ncols)))

    for i, pos in enumerate(plot_positions):
        img_path = os.path.join(sns_dir, selected_images[i])
        img = imread(img_path)
        axes[pos[0], pos[1]].imshow(img)
        axes[pos[0], pos[1]].axis("off")
        axes[pos[0], pos[1]].set_title(f"{img.shape[1]}x{img.shape[0]} px")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.pyplot(fig)
    return buf
