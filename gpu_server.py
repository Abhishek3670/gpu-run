import base64
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
AI_MODELS_PATH = Path(
    os.environ.get("AI_MODELS_PATH", str(BASE_DIR / "models"))
).expanduser()
YOLO_MODEL_NAME = os.environ.get("YOLO_MODEL_NAME", "yolov8s.pt")
YOLO_MODEL_PATH = AI_MODELS_PATH / YOLO_MODEL_NAME
KNOWN_FACES_DIR = AI_MODELS_PATH / "known_faces"
KNOWN_ENCODINGS_PATH = AI_MODELS_PATH / "known_faces_encodings.pkl"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU server starting on device: {device}")
print(f"Models path: {AI_MODELS_PATH}")


def resolve_model_path(loaded_model):
    if YOLO_MODEL_PATH.exists():
        return YOLO_MODEL_PATH

    ckpt_path = getattr(loaded_model, "ckpt_path", None)
    if ckpt_path:
        candidate = Path(ckpt_path)
        if candidate.exists():
            return candidate

    return YOLO_MODEL_PATH


def load_yolo_model():
    AI_MODELS_PATH.mkdir(parents=True, exist_ok=True)

    model_source = YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else YOLO_MODEL_NAME
    if not YOLO_MODEL_PATH.exists():
        print(f"YOLO model not found at {YOLO_MODEL_PATH}")
        print("Downloading YOLO weights on first run...")

    loaded_model = YOLO(str(model_source))
    loaded_model.to(device)

    if not YOLO_MODEL_PATH.exists():
        candidate_paths = []
        ckpt_path = getattr(loaded_model, "ckpt_path", None)
        if ckpt_path:
            candidate_paths.append(Path(ckpt_path))
        candidate_paths.append(Path.cwd() / YOLO_MODEL_NAME)

        for candidate in candidate_paths:
            if candidate.exists() and candidate.resolve() != YOLO_MODEL_PATH.resolve():
                shutil.copy2(candidate, YOLO_MODEL_PATH)
                break

    return loaded_model


try:
    model = load_yolo_model()
    ACTIVE_MODEL_PATH = resolve_model_path(model)
    model_size_mb = (
        ACTIVE_MODEL_PATH.stat().st_size / (1024 * 1024)
        if ACTIVE_MODEL_PATH.exists()
        else 0
    )
    print("YOLO model loaded successfully")
    print(f"  Path: {ACTIVE_MODEL_PATH}")
    if model_size_mb:
        print(f"  Size: {model_size_mb:.1f} MB")
    print(f"  Device: {device}")
except Exception as e:
    print(f"FATAL: Error loading YOLO model: {e}")
    print("Set AI_MODELS_PATH to a writable directory if needed.")
    raise SystemExit(1)

try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Cascade classifier could not be loaded.")
    print("OpenCV face cascade loaded successfully")
except Exception as e:
    print(f"ERROR: Could not load face cascade: {e}")
    face_cascade = None

known_face_encodings = []
known_face_names = []
print("Enhanced face detection enabled: YOLO person detection + OpenCV face detection")
print("Face recognition is disabled because face_recognition is not configured.")


def get_model_info():
    try:
        model_path = ACTIVE_MODEL_PATH if ACTIVE_MODEL_PATH.exists() else YOLO_MODEL_PATH
        if model_path.exists():
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            return {
                "path": str(model_path),
                "size_mb": round(model_size_mb, 1),
                "exists": True,
                "classes_count": len(model.names) if "model" in globals() else "Unknown",
            }

        return {
            "path": str(model_path),
            "size_mb": 0,
            "exists": False,
            "classes_count": "Unknown",
        }
    except Exception as e:
        return {
            "path": str(YOLO_MODEL_PATH),
            "error": str(e),
            "exists": False,
        }


def decode_image(frame_b64):
    try:
        img_data = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


@app.route("/health", methods=["GET"])
def health_check():
    model_info = get_model_info()

    return jsonify(
        {
            "status": "healthy",
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "models": {
                "yolo": model_info,
                "face_cascade": {
                    "enabled": face_cascade is not None,
                    "type": "OpenCV Haar Cascade",
                },
                "face_encodings": {
                    "loaded": len(known_face_names),
                    "path": str(KNOWN_ENCODINGS_PATH),
                },
            },
            "paths": {
                "models_directory": str(AI_MODELS_PATH),
                "yolo_model": str(YOLO_MODEL_PATH),
                "known_faces": str(KNOWN_FACES_DIR),
                "encodings": str(KNOWN_ENCODINGS_PATH),
            },
            "detection_methods": {
                "object_detection": "YOLO v8",
                "person_detection": "YOLO v8",
                "face_detection": (
                    "Enhanced: YOLO Person + OpenCV Face"
                    if face_cascade is not None
                    else "Disabled"
                ),
                "face_recognition": "Disabled (face_recognition library not available)",
            },
            "capabilities": {
                "can_detect_objects": True,
                "can_detect_persons": True,
                "can_detect_faces": face_cascade is not None,
                "can_recognize_faces": False,
                "enhanced_face_detection": True,
            },
            "server_info": {
                "version": "2.0",
                "description": "Enhanced GPU Server with YOLO + OpenCV Face Detection",
            },
        }
    )


@app.route("/detect", methods=["POST"])
def detect_objects():
    try:
        content = request.get_json(silent=True) or {}
        if "frame" not in content:
            return jsonify({"success": False, "error": "No frame data provided"})

        frame = decode_image(content["frame"])
        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode image"})

        results = model(frame, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]

                    detections.append(
                        {
                            "class_name": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

        return jsonify({"success": True, "detections": detections})
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/recognize", methods=["POST"])
def recognize_faces():
    try:
        content = request.get_json(silent=True) or {}
        if "frame" not in content:
            return jsonify({"success": False, "error": "No frame data provided"})

        frame = decode_image(content["frame"])
        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode image"})

        detected_faces = []
        results = model(frame, verbose=False)
        persons_detected = False

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]

                    if class_name == "person":
                        persons_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        person_roi = frame[y1:y2, x1:x2]

                        if person_roi.size > 0 and face_cascade is not None:
                            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                            faces_in_person = face_cascade.detectMultiScale(
                                gray_roi,
                                scaleFactor=1.05,
                                minNeighbors=3,
                                minSize=(20, 20),
                            )

                            for (fx, fy, fw, fh) in faces_in_person:
                                face_x1 = x1 + fx
                                face_y1 = y1 + fy
                                face_x2 = face_x1 + fw
                                face_y2 = face_y1 + fh

                                detected_faces.append(
                                    {
                                        "name": "Person",
                                        "confidence": conf * 0.9,
                                        "bbox": [face_x1, face_y1, face_x2, face_y2],
                                        "detection_method": "yolo_person + opencv_face",
                                        "person_confidence": conf,
                                        "person_bbox": [x1, y1, x2, y2],
                                    }
                                )

        if not persons_detected and face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            for (x, y, w, h) in faces:
                detected_faces.append(
                    {
                        "name": "Unknown",
                        "confidence": 0.75,
                        "bbox": [x, y, x + w, y + h],
                        "detection_method": "opencv_cascade_direct",
                    }
                )

        return jsonify(
            {
                "success": True,
                "faces": detected_faces,
                "persons_detected": persons_detected,
                "detection_methods_used": [
                    "yolo_person_detection",
                    "opencv_face_detection",
                ],
                "note": (
                    "Using enhanced YOLO + OpenCV detection. "
                    "For identity recognition, face_recognition is still needed."
                ),
            }
        )
    except Exception as e:
        print(f"Face detection error: {e}")
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("=" * 60)
    print("GPU SERVER FOR CCTV AI SYSTEM")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Models Path: {AI_MODELS_PATH}")
    print(f"Face Encodings Loaded: {len(known_face_names)}")
    print("=" * 60)
    print(f"Starting server on http://{HOST}:{PORT}")
    print("Endpoints:")
    print("  GET  /health     - Health check")
    print("  POST /detect     - Object detection")
    print("  POST /recognize  - Face recognition")
    print("=" * 60)

    app.run(host=HOST, port=PORT, debug=False, threaded=True)
