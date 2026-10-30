# Avocado Backend

FastAPI service that runs RF-DETR inference for avocado ripeness classification.

## Prerequisites

- Docker (recommended) — or a local Python 3.10 venv with the deps installed
- The model weights `model_state_dict.pth` (download or obtain separately)

## Run locally with Docker (recommended)

1. Build the image:
   ```bash
   docker build -t avocado-api-test .
   ```

2. Run it, mounting the fine-tuned weights:
   ```bash
   docker run --rm -p 8080:8080 \
     -e MODEL_PATH=/tmp/model_state_dict.pth \
     -e MODEL_SHA256=c3519cb25d54e4b4f72e81b26090db29f5e476ef80cf34bd0d28ff563e0d9960 \
     -v "$PWD/model_state_dict.pth:/tmp/model_state_dict.pth:ro" \
     avocado-api-test
   ```

3. Check it's up:
   ```bash
   curl localhost:8080/
   # => {"status":"Avocado API is running","model_loaded":false}
   ```
   (The model loads lazily on the first `/predict` request.)

## Run locally without Docker (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

MODEL_PATH="$PWD/model_state_dict.pth" \
MODEL_SHA256=c3519cb25d54e4b4f72e81b26090db29f5e476ef80cf34bd0d28ff563e0d9960 \
uvicorn main:app --host 0.0.0.0 --port 8080
```

## Call the prediction endpoint

`/predict` requires a valid Clerk JWT in `Authorization: Bearer <token>` for real use.

For quick manual testing of the *model* (bypassing auth), run inference directly:

```bash
docker exec <container_name> python -c "
from PIL import Image
from app.inference import predict_images
img = Image.open('/tmp/test.jpg').convert('RGB')
print(predict_images([img]))
"
```

## Configuration

See `.env.example` for all environment variables. Multi-image: `/predict`
accepts up to 4 `files` parts and aggregates them by median.
