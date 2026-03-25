# GPU Server

Lightweight Flask service for CCTV-style inference workflows. The main server in
`gpu_server.py` exposes:

- `GET /health` for runtime and model status
- `POST /detect` for YOLO object detection
- `POST /recognize` for person-plus-face detection using YOLO and OpenCV

## Repo Layout

- `gpu_server.py`: main Flask application
- `app.py`: small local helper script for scanning labels from JSON files

The repository intentionally ignores local virtual environments, downloaded
model weights, and temporary editor files so Git only tracks source and project
metadata.

## Setup

1. Create and activate a virtual environment.
2. Install a matching PyTorch build for your CPU or CUDA setup from the official
   PyTorch instructions.
3. Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The server reads these optional environment variables:

- `AI_MODELS_PATH`: directory that stores YOLO weights and face data
- `YOLO_MODEL_NAME`: model filename to load, defaults to `yolov8s.pt`
- `HOST`: Flask bind host, defaults to `0.0.0.0`
- `PORT`: Flask bind port, defaults to `5000`

If `AI_MODELS_PATH` is not set, the server uses a local `models/` directory
under the repo root and downloads the YOLO weights on first run.

## Run

```bash
python gpu_server.py
```

## Notes

- The current `/recognize` endpoint performs enhanced detection, not identity
  recognition. The `face_recognition` dependency is not wired in.
- Downloaded weights and other runtime artifacts are intentionally excluded from
  Git.
