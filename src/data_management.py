import os
import urllib.request
import joblib
from tensorflow.keras.models import load_model

# Google Drive direct download URL for your model file
MODEL_URL = "https://drive.google.com/uc?export=download&id=1GYpG0YDaNUGaxTx5c4CgVp5rbksmiptP"


def download_if_missing(path, url):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading model from {url} ...")
        urllib.request.urlretrieve(url, path)


def load_model_and_metadata(model_path, image_shape_path, class_index_path):
    # Download model if missing
    download_if_missing(model_path, MODEL_URL)

    # Load model and metadata
    model = load_model(model_path)
    image_shape = joblib.load(image_shape_path)
    class_indices = joblib.load(class_index_path)
    return model, image_shape, class_indices


def load_class_mapping(class_indices):
    return {v: k for k, v in class_indices.items()}
