"""RF-DETR model loading with integrity verification."""
import hashlib
import hmac
import logging
import os
import threading
from pathlib import Path

import torch

from app import config

logger = logging.getLogger(__name__)

_weights_filename = config.MODEL_PATH
pytorch_model = None
model_load_error = None
model_lock = threading.Lock()


def ensure_model_available() -> None:
    """Download the model from GCS to the local path if not already present."""
    if os.path.exists(_weights_filename):
        return
    if not config.MODEL_GCS_URI or not config.MODEL_GCS_URI.startswith("gs://"):
        raise RuntimeError("MODEL_GCS_URI must be a private gs:// URI")

    bucket_name, object_name = config.MODEL_GCS_URI[5:].split("/", 1)
    destination = Path(_weights_filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")

    try:
        from google.cloud import storage

        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_name)
        blob.download_to_filename(str(temporary_path))
        os.replace(temporary_path, destination)
        logger.info("Downloaded model artifact from Cloud Storage")
    except Exception:
        temporary_path.unlink(missing_ok=True)
        logger.exception("Model artifact download failed")
        raise


def sha256_file(path: str) -> str:
    model_hash = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            model_hash.update(chunk)
    return model_hash.hexdigest()


def get_model():
    """Load (once, lazily) and return the inference model."""
    global pytorch_model, model_load_error

    if pytorch_model is not None:
        return pytorch_model

    with model_lock:
        if pytorch_model is not None:
            return pytorch_model

        try:
            ensure_model_available()
            if not config.MODEL_SHA256:
                raise RuntimeError("MODEL_SHA256 is not configured")

            if not hmac.compare_digest(sha256_file(_weights_filename), config.MODEL_SHA256.lower()):
                raise RuntimeError("Model artifact hash does not match MODEL_SHA256")

            try:
                from rfdetr import RFDETRSmall
            except ImportError:
                import traceback

                print("RF-DETR import failed:")
                print(traceback.format_exc())
                raise

            base_model = RFDETRSmall(
                resolution=640,
                num_classes=1,
                num_attributes=1,
                download_dir=config.RFDETR_DOWNLOAD_DIR,
            )

            checkpoint = torch.load(
                _weights_filename,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            base_model.model.model.load_state_dict(state_dict)
            loaded_model = base_model.model.model
            loaded_model.eval()

            pytorch_model = loaded_model
            model_load_error = None
            logger.info("RFDETRSmall weights loaded successfully")
            return pytorch_model
        except Exception as e:
            import traceback

            model_load_error = traceback.format_exc()
            logger.exception("Model loading failed")
            raise


def model_is_loaded() -> bool:
    return pytorch_model is not None
