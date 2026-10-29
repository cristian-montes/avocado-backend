FROM python:3.10-slim

WORKDIR /app

ENV MPLCONFIGDIR=/tmp/matplotlib \
    RFDETR_DOWNLOAD_DIR=/tmp/rfdetr \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip==25.2 \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser

# Stage the base RF-DETR pretrain so RFDETRSmall() does not download it at runtime.
# The fork ignores download_dir and writes "rf-detr-small.pth" to the CWD (/app),
# which is non-writable for appuser — so we bake the file into the image instead.
# It is fully overwritten by the fine-tuned state_dict at load time, but is
# required by Model.__init__ (which dereferences checkpoint_num_classes unconditionally).
COPY rf-detr-small.pth /app/rf-detr-small.pth

COPY --chown=appuser:appuser . .

USER appuser

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]