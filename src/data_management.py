import os
import joblib
from tensorflow.keras.models import load_model


def load_model_and_metadata(model_path, image_shape_path, class_index_path):
    model = load_model(model_path)
    image_shape = joblib.load(image_shape_path)
    class_indices = joblib.load(class_index_path)
    return model, image_shape, class_indices


def load_class_mapping(class_indices):
    return {v: k for k, v in class_indices.items()}