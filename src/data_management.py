import os
import joblib
import gdown
from tensorflow.keras.models import load_model

# Google Drive File IDs
MODEL_ID = "1GYpG0YDaNUGaxTx5c4CgVp5rbksmiptP"
IMAGE_SHAPE_ID = "1eAlTBXjbEh6BslNDvTV3CqrECcFeAW8j"
CLASS_INDICES_ID = "1yLJPT2gQpijFCXzE9ogl6Desrmteka2e"

# Local paths
MODEL_PATH = "outputs/v1/cherry_leaf_mildew_model.h5"
IMAGE_SHAPE_PATH = "outputs/02_data_visualisation/image_shape.pkl"
CLASS_INDEX_PATH = "outputs/03_modelling_and_evaluating/class_indices.pkl"


def download_if_missing(path, file_id):
    # Always delete and re-download on Heroku
    if "DYNO" in os.environ and os.path.exists(path):
        print(f"Deleting existing file on Heroku to force re-download: {path}")
        os.remove(path)

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {os.path.basename(path)} from Google Drive...")
        gdown.download(url, path, quiet=False, fuzzy=True)
        print(f"Finished downloading: {path}")
        print(f"File size: {os.path.getsize(path)} bytes")


def load_model_and_metadata(
    model_path=MODEL_PATH,
    image_shape_path=IMAGE_SHAPE_PATH,
    class_index_path=CLASS_INDEX_PATH
):
    # Download model and metadata files if missing
    download_if_missing(model_path, MODEL_ID)
    download_if_missing(image_shape_path, IMAGE_SHAPE_ID)
    download_if_missing(class_index_path, CLASS_INDICES_ID)

    # Final pre-load logs
    print(f"About to load model from: {model_path}")
    print(f"Model file size: {os.path.getsize(model_path)} bytes")

    # Load the files
    model = load_model(model_path)
    image_shape = joblib.load(image_shape_path)
    class_indices = joblib.load(class_index_path)

    return model, image_shape, class_indices


def load_class_mapping(class_indices):
    return {v: k for k, v in class_indices.items()}
