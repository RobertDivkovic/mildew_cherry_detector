import joblib
from pathlib import Path

# Match the alphabetical mapping used by Keras flow_from_directory
class_indices = {"healthy": 0, "powdery_mildew": 1}

out_path = Path("outputs/03_modelling_and_evaluating/class_indices.pkl")
out_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(class_indices, out_path.as_posix())
print("Saved class_indices.pkl:", class_indices)
