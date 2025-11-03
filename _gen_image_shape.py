import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

model_path = r"outputs/v1/cherry_leaf_mildew_model.h5"
out_path = Path(r"outputs/02_data_visualisation/image_shape.pkl")
out_path.parent.mkdir(parents=True, exist_ok=True)

model = load_model(model_path)          # loads your .h5
image_shape = model.input_shape[1:]     # (H, W, C)
joblib.dump(image_shape, out_path.as_posix())
print("Saved image_shape.pkl with:", image_shape)
