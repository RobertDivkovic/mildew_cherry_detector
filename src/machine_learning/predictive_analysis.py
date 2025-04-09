import numpy as np
from tensorflow.keras.preprocessing import image


def preprocess_uploaded_image(uploaded_file, image_shape):
    img = image.load_img(uploaded_file, target_size=image_shape[:2])
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0


def predict_from_array(model, img_array, class_indices):
    prob = model.predict(img_array)[0, 0]
    class_map = {v: k for k, v in class_indices.items()}
    prediction = class_map[int(prob > 0.5)]

    if prediction == class_map[0]:
        prob = 1 - prob

    return prediction, prob