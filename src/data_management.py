import os
import joblib
import gdown
from tensorflow.keras.models import load_model

# Google Drive File ID (not full URL)
FILE_ID = "1GYpG0YDaNUGaxTx5c4CgVp5rbksmiptP"
MODEL_PATH = "outputs/v1/cherry_leaf_mildew_model.h5"


def download_if_missing(path, file_id):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"🔽 Downloading model from Google Drive using gdown...")
        gdown.download(url, path, quiet=False)


def load_model_and_metadata(model_path=MODEL_PATH, image_shape_path=None, class_index_path=None):
    # Download model file if missing
    download_if_missing(model_path, FILE_ID)

    # Load model and metadata
    model = load_model(model_path)
    image_shape = joblib.load(image_shape_path) if image_shape_path else None
    class_indices = joblib.load(class_index_path) if class_index_path else None
    return model, image_shape, class_indices


def load_class_mapping(class_indices):
    return {v: k for k, v in class_indices.items()}
