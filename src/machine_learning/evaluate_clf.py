import joblib
from tensorflow.keras.models import load_model


def evaluate_model(model_path, test_set, output_path=None):
    model = load_model(model_path)
    loss, accuracy = model.evaluate(test_set)

    if output_path:
        joblib.dump((loss, accuracy), output_path)

    return {"loss": loss, "accuracy": accuracy}
