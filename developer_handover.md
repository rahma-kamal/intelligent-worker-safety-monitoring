# Developer Handover & Integration Guide
## Intelligent Human Activity Recognition System for Factory Safety

This guide provides a comprehensive overview of the system architecture, folder structure, and code integration patterns to help developers integrate the tracking, action recognition, and safety reporting models into other applications.

---

## 📂 Project Directory Structure

Share this entire directory with the developer. It contains the following modules:

```text
factory_worker_safety_system/
│
├── config.yaml                    # System thresholds, restricted zones, and model parameters
├── requirements.txt               # Global dependencies (PyTorch, Ultralytics, FastAPI, etc.)
├── developer_handover.md          # This integration documentation
│
├── checkpoints/                   # Trained deep learning model weights (.pt files)
│   ├── har_bilstm_best.pt         # Bidirectional LSTM weights (2.6 MB)
│   ├── har_tcn_best.pt            # Temporal Convolutional Network weights (918 KB)
│   └── har_transformer_best.pt    # Spatial-Temporal Attention Transformer weights (349 KB)
│
├── core/                          # Core tracking and processing engines
│   ├── tracker.py                 # YOLOv8 worker detection and persistent tracking (ByteTrack)
│   ├── pose.py                    # HUD rendering and skeleton drawing engine
│   ├── safety_engine.py           # Posture checker and hybrid AI safety alert rules
│   └── pipeline.py                # Main orchestrator linking tracking -> pose -> HAR -> alerts
│
├── models/                        # PyTorch Temporal HAR Neural Network Architectures
│   ├── base_har.py                # Abstract base model wrapper (loading/saving states)
│   ├── bilstm.py                  # Bidirectional LSTM network
│   ├── tcn.py                     # Temporal Convolutional Network (1D causal convolutions)
│   └── transformer.py             # Spatial-Temporal Multi-Head Attention Transformer
│
├── data/                          # Data pipelines and feature preprocessing
│   ├── preprocess.py              # Normalized skeletal coordinate and joint angle extractor
│   ├── dataset.py                 # PyTorch Dataset and loader with data augmentations
│   └── extract_dataset_keypoints.py # Feature extractor to compile video clips into .npy files
│
├── reporting/                     # Safety reports exporter
│   └── generator.py               # Generates frame-accurate CSV logs, JSON, and dark-mode HTML reports
│
└── templates/                     # Live streaming dashboard frontend
    ├── index.html                 # Glassmorphic UI with WebSocket visual skeleton rendering
    └── style.css                  # UI dark-theme styling sheet
```

---

## 🚀 How to Load and Use the Trained Models

The trained action recognition models are saved as standard PyTorch state checkpoints under the `checkpoints/` folder. Below is a minimal integration script showing how to use the pipeline in standard Python applications:

```python
import cv2
from core.pipeline import WorkerMonitoringPipeline
from reporting.generator import SafetyReportGenerator

# 1. Initialize the pipeline with your chosen model architecture
# The pipeline automatically manages YOLOv8 tracking and loads the best trained checkpoints.
pipeline = WorkerMonitoringPipeline(
    model_type="tcn",                             # Choices: "bilstm", "tcn", "transformer"
    checkpoint_path="checkpoints/har_tcn_best.pt", # Path to trained .pt weights
    config_path="config.yaml"
)

# 2. Initialize the safety report generator
report_gen = SafetyReportGenerator()

# 3. Open your video source (can be a local video file, IP stream, RTSP, or camera index)
cap = cv2.VideoCapture("data/workplace_dataset/sample_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_idx += 1
    
    # 4. Feed the raw frame into the pipeline
    # Returns the annotated frame (with drawn skeletons/HUDs) and a list of worker events
    annotated_frame, events = pipeline.process_frame(
        frame=frame,
        report_generator=report_gen,
        frame_idx=frame_idx,
        fps=fps
    )
    
    # 5. Retrieve real-time data for the current frame
    for event in events:
        track_id = event["track_id"]
        predicted_activity = event["ai_activity"]
        risk_score = event["risk_score"]
        risk_level = event["risk_level"]         # "Low", "Medium", or "High"
        alert_type = event["alert_type"]         # e.g., "Walkway Boundary Crossing" or "None"
        
        print(f"Frame: {frame_idx} | Worker #{track_id} | Activity: {predicted_activity} | Alert: {alert_type}")
        
    # Display the processed frame (optional)
    cv2.imshow("Worker Safety Stream", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 6. Export post-session analytics reports
report_gen.export_csv("reports/final_report.csv")
report_gen.export_json("reports/final_report.json")
report_gen.export_html_report("reports/final_report.html")
```

---

## 🌐 FastAPI & WebSocket Real-Time Streaming API

If the developer is building a web, mobile, or desktop application, they can use the FastAPI server (`main.py`) to stream real-time annotated frames and telemetry.

### 1. WebSocket Endpoint (`ws://127.0.0.1:8000/ws`)
When a client connects to this endpoint, the backend starts decoding frames, runs the tracking pipeline, and streams JSON packets at ~24 FPS.

#### Outbound JSON Packet Format:
```json
{
  "frame": "/9j/4AAQSkZJRgABAQAAAQABAAD...",  // Base64 encoded JPEG frame
  "active_workers": [
    {
      "track_id": 1,
      "activity": "Authorized Intervention",
      "risk_score": 18.2,
      "risk_level": "Low",
      "alert": "None"
    },
    {
      "track_id": 2,
      "activity": "Safe Walkway Violation",
      "risk_score": 82.0,
      "risk_level": "High",
      "alert": "Walkway Boundary Crossing"
    }
  ],
  "telemetry": {
    "productivity_index": 78.5,
    "safety_index": 85.0,
    "violations_count": 2,
    "average_risk": 41.2
  }
}
```

### 2. HTTP Reports API (`GET http://127.0.0.1:8000/api/reports`)
Lists all available HTML, CSV, and JSON safety reports generated on the server:
```json
[
  {
    "filename": "safety_report_20260611_121014.html",
    "url": "/reports/safety_report_20260611_121014.html",
    "size_kb": 9.1,
    "type": "HTML"
  }
]
```

---

## ⚙️ Dependencies

Ensure the target runtime environment has the requirements installed:
```bash
pip install -r requirements.txt
```
*Key libraries: `torch`, `torchvision`, `ultralytics` (YOLOv8), `opencv-python` (video decoding), `fastapi` and `uvicorn` (server & websockets).*
