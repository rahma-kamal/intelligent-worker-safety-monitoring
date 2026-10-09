import os
import cv2
import json
import base64
import asyncio
import time
import yaml
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core.pipeline import WorkerMonitoringPipeline
from reporting.generator import SafetyReportGenerator

app = FastAPI(title="Intelligent Worker Safety Monitoring Platform")

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Mount static and reports directory
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# Load config
with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Global orchestrators (lazy-loaded when WebSocket connects)
pipeline = None
report_gen = None

@app.get("/")
async def get_dashboard():
    """
    Serves the primary premium dashboard HTML frontend.
    """
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(html_path):
        return HTMLResponse(content=open(html_path, "r", encoding="utf-8").read(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard Template Not Found. Please implement templates/index.html</h1>", status_code=404)

@app.get("/api/reports")
async def list_reports():
    """
    Lists all generated CSV, JSON, and HTML reports.
    """
    files = []
    if os.path.exists(REPORTS_DIR):
        for f in os.listdir(REPORTS_DIR):
            if f.endswith(('.html', '.csv', '.json')):
                path = f"/reports/{f}"
                size = os.path.getsize(os.path.join(REPORTS_DIR, f))
                files.append({
                    "filename": f,
                    "url": path,
                    "size_kb": round(size / 1024, 1),
                    "type": f.split('.')[-1].upper()
                })
    return sorted(files, key=lambda x: x["filename"], reverse=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles active WebSocket streams.
    Decodes demo video frame-by-frame, runs tracking, keypoints, and HAR,
    then broadcasts base64 JPEGs and analytics telemetry in real time.
    """
    global pipeline, report_gen
    await websocket.accept()
    print("[*] WebSocket connection established with client dashboard.")
    
    # 1. Initialize pipeline and reports
    if pipeline is None:
        # Load best trained BiLSTM checkpoint if it exists, otherwise fallback to default
        ckpt_path = os.path.join(BASE_DIR, "checkpoints", "har_bilstm_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = None
        pipeline = WorkerMonitoringPipeline(model_type="bilstm", checkpoint_path=ckpt_path)
    
    report_gen = SafetyReportGenerator()
    
    # 2. Check for video source
    video_source = None
    raw_video_dir = os.path.join(BASE_DIR, config['dataset']['raw_video_dir'])
    os.makedirs(raw_video_dir, exist_ok=True)
    
    videos = [f for f in os.listdir(raw_video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
    if videos:
        video_source = os.path.join(raw_video_dir, videos[0])
        print(f"[*] Serving video stream from local file: '{video_source}'")
    else:
        print("[!] No physical video files found in workplace_dataset directory. Running in HIGH-TECH SIMULATION STREAM MODE.")
        
    cap = None
    if video_source:
        cap = cv2.VideoCapture(video_source)
        
    # Simulated frame loop counter
    sim_t = 0
    
    try:
        while True:
            annotated_frame = None
            events = []
            
            # --- CASE A: Physical Video Capture Active ---
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Downscale for smooth WebSocket transmission
                h, w = frame.shape[:2]
                target_w = 854  # 480p standard for visual dashboard efficiency
                target_h = int(h * (target_w / w))
                frame_resized = cv2.resize(frame, (target_w, target_h))
                
                # Execute pipeline processing
                annotated_frame, events = pipeline.process_frame(frame_resized, report_generator=report_gen)
                
            # --- CASE B: Simulation Stream Fallback ---
            else:
                # If no video is present, we render a highly visual, realistic mock monitoring feed
                # complete with drawing dynamic workers, moving skeletons, grid lines, and safety bounds
                sim_t += 1
                sim_frame = np.zeros((480, 854, 3), dtype=np.uint8)
                
                # Draw grid patterns representing safety environment
                for grid_x in range(0, 854, 40):
                    cv2.line(sim_frame, (grid_x, 0), (grid_x, 480), (12, 16, 22), 1)
                for grid_y in range(0, 480, 40):
                    cv2.line(sim_frame, (0, grid_y), (854, grid_y), (12, 16, 22), 1)
                    
                # Render restricted zones (bounding lines)
                cv2.rectangle(sim_frame, (100, 100), (350, 400), (0, 0, 80), 2)
                cv2.putText(sim_frame, "RESTRICTED AREA G-3", (110, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
                
                # Simulate a worker keypoint array
                # Worker 1: Standing in panel area
                w1_kps = [
                    [427 + 30 * np.sin(sim_t * 0.1), 180 + 5 * np.cos(sim_t * 0.15), 0.9], # Nose
                    # Torso & limbs
                    [400, 220, 0.9], [450, 220, 0.9], # shoulders
                    [380 + 20 * np.sin(sim_t * 0.2), 270, 0.9], [470 + 20 * np.cos(sim_t * 0.2), 270, 0.9], # elbows
                    [370, 310 + 20 * np.cos(sim_t * 0.3), 0.9], [480, 310 + 20 * np.sin(sim_t * 0.3), 0.9], # wrists
                    [410, 320, 0.9], [440, 320, 0.9], # hips
                    [410, 380, 0.9], [440, 380, 0.9], # knees
                    [410, 440, 0.9], [440, 440, 0.9]  # ankles
                ]
                # Pad to 17 keypoints
                w1_full_kps = np.zeros((17, 3))
                w1_full_kps[0:15] = w1_kps[0:15]
                
                # Run simulated safety evaluation
                # Worker 1: Normal intervention
                w1_activity = "Authorized_Intervention"
                w1_alert = "None"
                w1_risk_level = "Low"
                w1_risk_score = 18.0 + 5 * np.sin(sim_t * 0.05)
                
                # Worker 2: Bending or Falling
                w2_y_offset = 240 + 100 * np.sin(sim_t * 0.02) # Moves around
                is_falling = (sim_t % 300) > 230  # Periodic fall simulation
                
                if is_falling:
                    w2_activity = "Walking"
                    w2_alert = "Worker Fall Detected"
                    w2_risk_level = "High"
                    w2_risk_score = 98.0
                    w2_kps = [
                        [250, 380, 0.9], [290, 370, 0.9], [290, 390, 0.9],
                        [320, 360, 0.9], [320, 400, 0.9], [340, 360, 0.9], [340, 400, 0.9],
                        [350, 375, 0.9], [350, 385, 0.9], [400, 370, 0.9], [400, 390, 0.9],
                        [440, 370, 0.9], [440, 390, 0.9], [440, 370, 0.9], [440, 390, 0.9]
                    ]
                else:
                    # Bending Carrying overload
                    is_overload = (sim_t % 300) > 120
                    if is_overload:
                        w2_activity = "Carrying_Overload"
                        w2_alert = "Ergonomic Lifting Threat"
                        w2_risk_level = "High"
                        w2_risk_score = 82.0
                        w2_kps = [
                            [250, 240, 0.9], [230, 270, 0.9], [270, 270, 0.9],
                            [210, 300, 0.9], [290, 300, 0.9], [220, 330, 0.9], [280, 330, 0.9],
                            [235, 340, 0.9], [265, 340, 0.9], [235, 390, 0.9], [265, 390, 0.9],
                            [235, 440, 0.9], [265, 440, 0.9], [235, 440, 0.9], [265, 440, 0.9]
                        ]
                    else:
                        w2_activity = "Safe_Walkway"
                        w2_alert = "None"
                        w2_risk_level = "Low"
                        w2_risk_score = 12.0
                        w2_kps = [
                            [250, 200, 0.9], [235, 230, 0.9], [265, 230, 0.9],
                            [220, 270, 0.9], [280, 270, 0.9], [220, 310, 0.9], [280, 310, 0.9],
                            [235, 320, 0.9], [265, 320, 0.9], [235, 370, 0.9], [265, 370, 0.9],
                            [235, 420, 0.9], [265, 420, 0.9], [235, 420, 0.9], [265, 420, 0.9]
                        ]
                        
                w2_full_kps = np.zeros((17, 3))
                w2_full_kps[0:15] = w2_kps[0:15]
                
                # Render Skeletons & Bounding Boxes manually for simulation
                # Draw worker 1
                pipeline.visualizer.draw_skeleton(sim_frame, w1_full_kps, COLORS['joint'])
                pipeline.visualizer.draw_hud_card(
                    sim_frame, [350, 150, 500, 450], 1,
                    w1_activity, w1_risk_level, w1_risk_score, w1_alert
                )
                events.append({
                    "track_id": 1, "ai_activity": w1_activity,
                    "risk_level": w1_risk_level, "risk_score": w1_risk_score, "alert_type": w1_alert
                })
                report_gen.add_record(1, w1_activity, w1_alert != "None", w1_alert, w1_risk_score, 12.0, 0.4)
                
                # Draw worker 2
                pipeline.visualizer.draw_skeleton(sim_frame, w2_full_kps, COLORS['joint'] if w2_risk_level=="Low" else COLORS['danger'])
                w2_box = [180, 330, 470, 460] if is_falling else [190, 170, 310, 450]
                pipeline.visualizer.draw_hud_card(
                    sim_frame, w2_box, 2,
                    w2_activity, w2_risk_level, w2_risk_score, w2_alert
                )
                events.append({
                    "track_id": 2, "ai_activity": w2_activity,
                    "risk_level": w2_risk_level, "risk_score": w2_risk_score, "alert_type": w2_alert
                })
                report_gen.add_record(2, w2_activity, w2_alert != "None", w2_alert, w2_risk_score, 45.0, 0.8)
                
                annotated_frame = sim_frame
                
            # 3. Encode processed annotated frame to base64 JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 4. Compile dynamic telemetry statistics
            stats = report_gen.get_summary_statistics()
            
            websocket_packet = {
                "frame": frame_base64,
                "active_workers": [
                    {
                        "track_id": int(e["track_id"]),
                        "activity": e["ai_activity"].replace('_', ' '),
                        "risk_score": round(float(e["risk_score"]), 1),
                        "risk_level": e["risk_level"],
                        "alert": e["alert_type"]
                    }
                    for e in events
                ],
                "telemetry": {
                    "productivity_index": stats.get("productivity_index", 100.0),
                    "safety_index": stats.get("safety_index", 100.0),
                    "violations_count": stats.get("safety_violations_count", 0),
                    "average_risk": stats.get("average_risk_score", 0.0)
                }
            }
            
            # Send packet
            await websocket.send_json(websocket_packet)
            
            # Export reports periodically to keep downloadable logs up to date
            if time.time() % 30 < 0.5:
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                report_gen.export_html_report(os.path.join(REPORTS_DIR, f"live_report_{timestamp}.html"))
                
            # Control frame rate (~24 FPS)
            await asyncio.sleep(0.04)
            
    except WebSocketDisconnect:
        print("[!] Client disconnected from WebSocket.")
        if cap:
            cap.release()
            
    except Exception as e:
        print(f"[!] Stream error: {e}")
        if cap:
            cap.release()

if __name__ == "__main__":
    import uvicorn
    # Execute uvicorn server
    uvicorn.run(
        "main:app",
        host=config['server']['host'],
        port=config['server']['port'],
        log_level=config['server']['log_level']
    )
