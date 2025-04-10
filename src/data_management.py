import os
import joblib
import gdown
import zipfile
from tensorflow.keras.models import load_model

# Google Drive File IDs
MODEL_ID = "1GYpG0YDaNUGaxTx5c4CgVp5rbksmiptP"
IMAGE_SHAPE_ID = "1eAlTBXjbEh6BslNDvTV3CqrECcFeAW8j"
CLASS_INDICES_ID = "1yLJPT2gQpijFCXzE9ogl6Desrmteka2e"
POWDERY_MILDEW_ZIP_ID = "1lVl9K68iQ4X6METcyZtH8xqEq42zxcee"
HEALTHY_ZIP_ID = "1o1Q-iih2WCFrPYohXWevDAvm4zdeI7pH"

# Local paths
MODEL_PATH = "outputs/v1/cherry_leaf_mildew_model.h5"
IMAGE_SHAPE_PATH = "outputs/02_data_visualisation/image_shape.pkl"
CLASS_INDEX_PATH = "outputs/03_modelling_and_evaluating/class_indices.pkl"
POWDERY_ZIP_PATH = "inputs/powdery_mildew.zip"
HEALTHY_ZIP_PATH = "inputs/healthy.zip"
EXTRACTED_DATASET_PATH = "inputs"


def is_h5_valid(path):
    try:
        with open(path, 'rb') as f:
            signature = f.read(8)
            return signature == b'\x89HDF\r\n\x1a\n'
    except Exception as e:
        print(f"[ERROR] Could not validate HDF5 file at {path}: {e}")
        return False


def download_if_missing(path, file_id):
    if "DYNO" in os.environ and os.path.exists(path):
        print(f"[Heroku] Deleting existing file to force re-download: {path}")
        os.remove(path)

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[Download] Fetching {os.path.basename(path)} from Google Drive...")
        gdown.download(url, path, quiet=False, fuzzy=True)
        print(f"[Download] Finished downloading: {path}")
        print(f"[Download] File size: {os.path.getsize(path)} bytes")


def unzip_if_needed(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"[Unzip] Extracted {zip_path} to {extract_to}")


def load_model_and_metadata(
    model_path=MODEL_PATH,
    image_shape_path=IMAGE_SHAPE_PATH,
    class_index_path=CLASS_INDEX_PATH
):
    # Download model and metadata
    download_if_missing(model_path, MODEL_ID)
    download_if_missing(image_shape_path, IMAGE_SHAPE_ID)
    download_if_missing(class_index_path, CLASS_INDICES_ID)
    download_if_missing(POWDERY_ZIP_PATH, POWDERY_MILDEW_ZIP_ID)
    download_if_missing(HEALTHY_ZIP_PATH, HEALTHY_ZIP_ID)

    # Unzip datasets to correct directory (train folder)
    unzip_if_needed(POWDERY_ZIP_PATH, os.path.join(EXTRACTED_DATASET_PATH, "cherry_leaves_split", "train"))
    unzip_if_needed(HEALTHY_ZIP_PATH, os.path.join(EXTRACTED_DATASET_PATH, "cherry_leaves_split", "train"))

    # Check model integrity before loading
    if not is_h5_valid(model_path):
        raise OSError(f"[ERROR] File at '{model_path}' is not a valid .h5 file (possible corrupted download).")

    print(f"[Model] Loading model from: {model_path}")
    print(f"[Model] Model file size: {os.path.getsize(model_path)} bytes")

    # Load the files
    model = load_model(model_path)
    image_shape = joblib.load(image_shape_path)
    class_indices = joblib.load(class_index_path)

    return model, image_shape, class_indices


def load_class_mapping(class_indices):
    return {v: k for k, v in class_indices.items()}
