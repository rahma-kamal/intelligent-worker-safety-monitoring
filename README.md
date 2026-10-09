# Intelligent Worker Safety Monitoring Platform Using Deep Learning and Computer Vision

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8--Pose-00599C.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end industrial computer vision and deep learning platform designed for real-time factory worker monitoring, ergonomic hazard detection, temporal human activity recognition (HAR), and automated occupational safety compliance reporting.

---

## 📌 Overview & Key Capabilities

Industrial environments present high ergonomic risks, hazardous machinery zones, and safety compliance challenges. This platform continuously monitors CCTV or IP camera video streams using a multi-stage deep learning pipeline:

1. **Worker Detection & Multi-Object Tracking:** YOLOv8-Pose combined with ByteTrack provides persistent multi-worker tracking IDs across occlusions and dense industrial settings.
2. **Biomechanical & Skeletal Kinematics:** Extracts 17 body keypoints and computes 92-dimensional spatial-temporal feature vectors including joint angles (spine tilt, elbows, knees), normalized keypoint coordinates, aspect ratios, and velocity dynamics across sliding temporal windows (30 frames).
3. **Comparative Temporal HAR Architectures:** Implements and benchmarks three neural architectures for sequential Human Activity Recognition:
   - **Bidirectional LSTM (BiLSTM):** Captures forward and reverse temporal dependencies.
   - **Temporal Convolutional Network (TCN):** 1D dilated causal convolutions with residual blocks for parallel feature extraction without vanishing gradients.
   - **Spatial-Temporal Transformer:** Multi-head self-attention encoder with learnable positional encoding.
4. **Hybrid Safety Engine:** Merges deep learning activity predictions with biomechanical rule heuristics:
   - **Ergonomic Lumbar Strain:** Detects dangerous forward bending angles (>50° trunk tilt) during lifting tasks.
   - **Fall Detection:** Evaluates bounding box aspect ratio shifts (>1.8 W/H) for immediate horizontal worker detection.
   - **Operational Risk Scoring:** Computes dynamic real-time risk scores (0–100) and severity categorizations (`Low`, `Medium`, `High`).
5. **Real-Time Glassmorphic Web Dashboard:** FastAPI backend broadcasting annotated video frames and telemetry at ~24 FPS via WebSockets to a modern dark-theme monitoring UI.
6. **Automated Safety Audit Reporting:** Aggregates frame-by-frame infractions into discrete incidents using temporal gap state-machine filtering, exporting audit reports in **HTML**, **CSV**, and **JSON**.

---

## 🏗️ System Architecture

```text
                                  [ Video Input Stream ]
                             (RTSP / Camera / Industrial MP4)
                                             │
                                             ▼
                             [ YOLOv8-Pose + ByteTrack ]
                         (Persistent Worker IDs & 17 Keypoints)
                                             │
                                             ▼
                             [ Skeletal Kinematics Engine ]
                      (Normalized Coordinates + Joint Angles + Velocity)
                                   (92-dim Feature Vectors)
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          [ Temporal HAR Models ]                       [ Biomechanical Rules ]
      ┌───────────────────────────────┐               (Spine Angle > 50°, Fall AR > 1.8,
      │  • BiLSTM                     │                Restricted Zone Violations)
      │  • Temporal ConvNet (TCN)     │                             │
      │  • Attention Transformer      │                             │
      └──────────────┬────────────────┘                             │
                     └──────────────────────┬───────────────────────┘
                                            ▼
                             [ Hybrid Safety Engine ]
                     (Risk Level: Low/Medium/High | Score: 0-100)
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
          [ FastAPI WebSocket Server ]               [ Compliance Exporter ]
             • 24 FPS JPEG Streaming                    • Interactive HTML Report
             • Real-time KPI Dashboard                  • Frame-level CSV Logs
             • Live Ergonomic Warnings                  • Session JSON Telemetry
```

---

## 📂 Project Structure

```text
intelligent-worker-safety-monitoring/
│
├── config.yaml                    # Global hyperparameters, safety thresholds, and zones
├── requirements.txt               # Dependencies (PyTorch, Ultralytics, FastAPI, etc.)
├── developer_handover.md          # Technical integration and handover documentation
│
├── checkpoints/                   # Trained deep learning checkpoints & evaluation plots
│   ├── har_bilstm_best.pt         # Trained Bidirectional LSTM weights (2.51 MB)
│   ├── har_tcn_best.pt            # Trained Temporal Convolutional Network weights (897 KB)
│   ├── har_transformer_best.pt    # Trained Attention Transformer weights (341 KB)
│   ├── confusion_matrix_bilstm.png
│   ├── confusion_matrix_tcn.png
│   └── confusion_matrix_transformer.png
│
├── core/                          # Core processing modules
│   ├── tracker.py                 # YOLOv8 worker detection & ByteTrack multi-object tracking
│   ├── pose.py                    # HUD rendering and skeleton visualizer
│   ├── safety_engine.py           # Hybrid AI + biomechanical safety violation rules
│   └── pipeline.py                # Pipeline orchestrator (Tracking -> Pose -> HAR -> Alerts)
│
├── models/                        # Temporal neural network architectures
│   ├── base_har.py                # Abstract base model wrapper (checkpoint serialization)
│   ├── bilstm.py                  # Bidirectional LSTM with LayerNorm & temporal pooling
│   ├── tcn.py                     # Temporal Convolutional Network with 1D causal convolutions
│   └── transformer.py             # Spatial-Temporal Transformer with Positional Encodings
│
├── reporting/                     # Safety analytics and export engine
│   └── generator.py               # Frame-accurate CSV, JSON, and dark-theme HTML report builder
│
├── reports/                       # Pre-generated sample compliance reports (HTML, CSV, JSON)
├── templates/                     # Glassmorphic web dashboard frontend
│   ├── index.html                 # WebSocket streaming interface & live KPI meters
│   └── style.css                  # Industrial UI styling system
│
├── check_environment.py           # Hardware, CUDA, and environment profiling script
├── train.py                       # Training pipeline supporting BiLSTM, TCN, and Transformer
├── evaluate.py                    # Multi-model evaluation, confusion matrix generator & benchmark
├── run_pipeline.py                # CLI inference runner on video sources
├── main.py                        # FastAPI web application with WebSocket streaming
├── output_annotated.mp4           # Sample processed video with annotated skeletons & alerts
└── yolov8n-pose.pt                # YOLOv8 Nano Pose detector weights (6.52 MB)
```

---

## 🏷️ Monitored Activity Classes

The platform classifies 8 distinct industrial activity classes:

| Class Index | Activity Label | Operational Classification | Severity Level |
| :---: | :--- | :--- | :---: |
| **0** | `Safe_Walkway_Violation` | Walkway Boundary Crossing | Medium |
| **1** | `Unauthorized_Intervention` | Unauthorized Machine Access | Medium |
| **2** | `Opened_Panel_Cover` | Exposed Electrical Hazard | Medium |
| **3** | `Carrying_Overload` | Ergonomic Lifting Threat | High |
| **4** | `Safe_Walkway` | Normal Walkway Movement | Safe / Low |
| **5** | `Authorized_Intervention` | Permitted Machine Operation | Safe / Low |
| **6** | `Closed_Panel_Cover` | Secured Equipment Standard | Safe / Low |
| **7** | `Safe_Carrying` | Standard Material Handling | Safe / Low |

---

## 📊 Experimental Evaluation & Model Benchmarks

The architectures were benchmarked for validation accuracy, weighted F1-score, model parameter size, and inference speed (FPS):

| Architecture | Model Parameters | Validation Accuracy | Weighted F1 | Inference Latency | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BiLSTM** | ~500K | High | High | Low Latency | Real-time (>60 FPS) |
| **TCN (Dilated Causal)** | ~250K | High | High | Ultra-low Latency | High Throughput (>90 FPS) |
| **Transformer (Attention)** | ~180K | High | High | Moderate Latency | Real-time (>45 FPS) |

### Confusion Matrices
Publication-ready normalized confusion matrices for each architecture are preserved under the [`checkpoints/`](checkpoints/) directory:
- `checkpoints/confusion_matrix_bilstm.png`
- `checkpoints/confusion_matrix_tcn.png`
- `checkpoints/confusion_matrix_transformer.png`

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support recommended (CPU inference fully supported)

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/rahma-kamal/intelligent-worker-safety-monitoring.git
cd intelligent-worker-safety-monitoring
pip install -r requirements.txt
```

### 3. Verify Hardware & Deep Learning Environment
Run the built-in diagnostic tool to check CPU, RAM, CUDA availability, and tensor throughput:
```bash
python check_environment.py
```

---

## 💻 Usage & Execution

### Option A: Run Inference on Video Files (CLI)
Execute the pipeline on any local video file or camera device:
```bash
# Run using the trained TCN model
python run_pipeline.py --source output_annotated.mp4 --model tcn --output results.mp4

# Run using the trained BiLSTM model
python run_pipeline.py --source path/to/industrial_cctv.mp4 --model bilstm

# Run using the Attention Transformer
python run_pipeline.py --source 0 --model transformer
```
The script will output an annotated MP4 video along with session reports in `reports/`.

### Option B: Launch Live Web Dashboard & WebSocket Server
Start the FastAPI server:
```bash
python main.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
- Real-time video frames streamed at 24 FPS with HUD keypoint overlays.
- Real-time KPI telemetry (Safety Index, Productivity Index, Average Risk Score).
- Access past HTML and CSV compliance reports at `http://127.0.0.1:8000/api/reports`.

### Option C: Model Training & Evaluation
To retrain or benchmark models:
```bash
# Train a specific model architecture (bilstm / tcn / transformer)
python train.py --model tcn --epochs 50 --batch_size 32

# Evaluate all trained checkpoints and generate confusion matrices
python evaluate.py
```

---

## 🔒 Safety, Privacy & Exclusions

This repository is maintained in compliance with data privacy and security standards:
- **No Private Credentials:** Scanned for and confirmed free of API keys, authentication tokens, and secrets.
- **Lightweight Checkpoints:** All saved models and media files are under 7 MB, well below GitHub's 100 MB file limit.
- **Clean Git Tracking:** Bytecode (`__pycache__`), virtual environments, and temporary caches are filtered via `.gitignore`.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
