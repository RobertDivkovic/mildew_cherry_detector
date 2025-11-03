# src/data_management.py
import os
import shutil
import requests
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
def proj_path(*parts) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))

def _download_stream(url: str, dst_path: str, chunk_size: int = 1 << 20):
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for c in r.iter_content(chunk_size=chunk_size):
                if c:
                    f.write(c)
    return str(dst)

def _fetch_model_if_needed(model_path: str, *, hf_repo_id: str | None = None,
                           hf_filename: str | None = None, http_url: str | None = None) -> str:
    if os.path.exists(model_path):
        return model_path

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    if hf_repo_id and hf_filename:
        if not HF_AVAILABLE:
            raise RuntimeError("Install huggingface_hub or remove HF_REPO_ID/HF_FILENAME.")
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")  # optional for private repos
        local_cache = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, token=token)
        shutil.copy(local_cache, model_path)
        return model_path

    if http_url:
        _download_stream(http_url, model_path)
        return model_path

    raise FileNotFoundError(
        f"Model not found: {model_path}. Provide HF_REPO_ID+HF_FILENAME or MODEL_URL, "
        f"or place the .h5 file locally at that path."
    )

def load_model_and_metadata(model_path: str, image_shape_path: str, class_indices_path: str):
    model_path       = proj_path(model_path)       # expect .h5
    image_shape_path = proj_path(image_shape_path)
    class_indices_path = proj_path(class_indices_path)

    model_file = _fetch_model_if_needed(
        model_path,
        hf_repo_id=os.getenv("HF_REPO_ID"),
        hf_filename=os.getenv("HF_FILENAME"),
        http_url=os.getenv("MODEL_URL"),
    )

    if not os.path.exists(image_shape_path):
        raise FileNotFoundError(f"Missing image shape: {image_shape_path}")
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"Missing class indices: {class_indices_path}")

    model = load_model(model_file)  # loads .h5
    image_shape = joblib.load(image_shape_path)
    class_indices = joblib.load(class_indices_path)
    return model, image_shape, class_indices
