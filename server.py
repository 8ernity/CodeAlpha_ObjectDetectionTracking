"""
FastAPI Server for AI Vision Studio: Real-time Object Detection & Tracking.
CrimeRakshak UI Theme Edition.
"""

import os
import time
import json
import base64
import cv2
import numpy as np
from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.detector import ObjectDetector, COCO_CLASSES
from tracker.sort import Sort
from tracker.deep_sort_tracker import ByteSortTracker

app = FastAPI(title="CodeAlpha - Real-time Object Detection & Tracking", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# State
class PipelineState:
    def __init__(self):
        self.detector = ObjectDetector(model_name="yolov8n.pt", conf_thresh=0.45, iou_thresh=0.45)
        self.tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.35)

state = PipelineState()


class ConfigPayload(BaseModel):
    conf_thresh: Optional[float] = None
    iou_thresh: Optional[float] = None
    max_age: Optional[int] = None
    min_hits: Optional[int] = None
    allowed_classes: Optional[List[str]] = None


@app.get("/api/status")
def get_status():
    return {
        "status": "ready",
        "device": state.detector.device,
        "model": "yolov8n",
        "classes": COCO_CLASSES,
        "config": {
            "conf_thresh": state.detector.conf_thresh,
            "iou_thresh": state.detector.iou_thresh,
            "max_age": state.tracker.max_age,
            "min_hits": state.tracker.min_hits,
            "allowed_classes": state.detector.allowed_classes
        },
        "stats": {
            "total_tracks": state.tracker.total_tracks_count
        }
    }


@app.post("/api/config")
def update_config(cfg: ConfigPayload):
    if cfg.conf_thresh is not None or cfg.iou_thresh is not None or cfg.allowed_classes is not None:
        state.detector.set_config(
            conf_thresh=cfg.conf_thresh,
            iou_thresh=cfg.iou_thresh,
            allowed_classes=cfg.allowed_classes
        )
    if cfg.max_age is not None:
        state.tracker.max_age = int(cfg.max_age)
    if cfg.min_hits is not None:
        state.tracker.min_hits = int(cfg.min_hits)

    return {"status": "updated", "config": get_status()["config"]}


@app.post("/api/reset")
def reset_tracker():
    state.tracker.reset()
    return {"status": "reset_successful"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    dest_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(dest_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    return {"status": "success", "filename": file.filename, "path": f"/uploads/{file.filename}"}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get("action")

            if action == "process_frame":
                frame_data = message.get("image")
                if not frame_data:
                    continue

                if "," in frame_data:
                    frame_data = frame_data.split(",")[1]

                img_bytes = base64.b64decode(frame_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                detections, meta = state.detector.detect(frame)
                tracks = state.tracker.update(detections)

                track_results = []
                for trk in tracks:
                    x1, y1, x2, y2, track_id, class_id, score, vx, vy = trk
                    track_id = int(track_id)
                    class_id = int(class_id)
                    class_name = state.detector.class_names.get(class_id, f"class_{class_id}")

                    track_results.append({
                        "id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "score": round(float(score), 3),
                        "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                        "vx": round(float(vx), 2),
                        "vy": round(float(vy), 2)
                    })

                response = {
                    "timestamp": time.time(),
                    "meta": meta,
                    "frame_w": frame.shape[1],
                    "frame_h": frame.shape[0],
                    "tracks": track_results,
                    "active_count": len(track_results),
                    "total_tracks": state.tracker.total_tracks_count
                }
                await websocket.send_text(json.dumps(response))

            elif action == "reset":
                state.tracker.reset()
                response = {
                    "type": "reset_ack",
                    "total_tracks": 0,
                    "active_count": 0,
                    "tracks": []
                }
                await websocket.send_text(json.dumps(response))

            elif action == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Error: {e}")


# Mount sample_data and uploads static routes
if os.path.exists(SAMPLE_DIR):
    app.mount("/sample_data", StaticFiles(directory=SAMPLE_DIR), name="sample_data")
if os.path.exists(UPLOADS_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Mount web root
if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[Server] Starting VisionGrid on http://0.0.0.0:{port} ...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
