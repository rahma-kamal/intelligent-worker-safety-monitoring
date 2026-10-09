import os
import cv2
import argparse
import time
import yaml

from core.pipeline import WorkerMonitoringPipeline
from reporting.generator import SafetyReportGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="Run Factory Safety & HAR Inference Pipeline")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to video file, folder of videos, or camera index (e.g. 0)")
    parser.add_argument("--model", type=str, default="bilstm", choices=["bilstm", "tcn", "transformer"],
                        help="HAR sequence model architecture to use")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained HAR model weights checkpoint (e.g., checkpoints/har_bilstm_best.pt)")
    parser.add_argument("--output", type=str, default="output_annotated.mp4",
                        help="Filepath to save annotated result video")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Config loading
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Check if a custom video source is given, otherwise locate/generate mock or prompt
    source = args.source
    if not source:
        print("[!] No video source provided. Looking for a demo video or utilizing simulation mode...")
        # Create a mock source directory if needed
        os.makedirs(config['dataset']['raw_video_dir'], exist_ok=True)
        # Search for any video file in the directory
        videos = [f for f in os.listdir(config['dataset']['raw_video_dir']) if f.endswith(('.mp4', '.avi', '.mkv'))]
        if videos:
            source = os.path.join(config['dataset']['raw_video_dir'], videos[0])
            print(f"[i] Auto-detected video file: '{source}'")
        else:
            print("[i] No videos found in workplace_dataset directory.")
            print("[i] To execute on live video, place an industrial MP4 in 'data/workplace_dataset/' or run:")
            print("    python run_pipeline.py --source <path_to_video.mp4>")
            return
            
    # Try parsing camera index
    if source.isdigit():
        source = int(source)
        
    # 2. Checkpoint fallback
    checkpoint = args.checkpoint
    if not checkpoint:
        # Check standard checkpoints directory
        default_ckpt = os.path.join(config['training']['save_dir'], f"har_{args.model}_best.pt")
        if os.path.exists(default_ckpt):
            checkpoint = default_ckpt
            
    # 3. Load components
    print(f"\n[*] Initializing Workplace Safety Pipeline using model: '{args.model.upper()}'...")
    pipeline = WorkerMonitoringPipeline(
        model_type=args.model,
        checkpoint_path=checkpoint,
        config_path=args.config
    )
    
    report_gen = SafetyReportGenerator()
    
    # 4. Open Video Stream
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: '{source}'")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[SUCCESS] Opened video successfully: Resolution {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    print(f"[*] Processing video frame-by-frame and writing annotations to '{args.output}'...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process Frame
            annotated_frame, events = pipeline.process_frame(frame, report_generator=report_gen, frame_idx=frame_count, fps=fps)
            
            # Draw standard system-level telemetry on the top bar
            current_fps = frame_count / (time.time() - start_time)
            telemetry_text = f"HAR Safety Platform | Frame: {frame_count}/{total_frames} | Pipeline FPS: {current_fps:.1f} | Active Workers: {len(events)}"
            cv2.putText(annotated_frame, telemetry_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Write annotated frame
            writer.write(annotated_frame)
            
            # Print console progress
            if frame_count % 30 == 0 or frame_count == total_frames:
                active_violations = sum(1 for e in events if e["alert_triggered"])
                print(f"Processed Frame {frame_count:04d}/{total_frames:04d} | Active Workers: {len(events)} | Violations: {active_violations} | System FPS: {current_fps:.1f}")
                
    except KeyboardInterrupt:
        print("[!] Execution interrupted by user.")
        
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Processing finished in {elapsed:.1f} seconds. Average System FPS: {frame_count / elapsed:.1f}")
    
    # 5. Export Safety Reports
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    csv_path = os.path.join(report_dir, f"safety_report_{timestamp}.csv")
    json_path = os.path.join(report_dir, f"safety_report_{timestamp}.json")
    html_path = os.path.join(report_dir, f"safety_report_{timestamp}.html")
    
    report_gen.export_csv(csv_path)
    report_gen.export_json(json_path)
    report_gen.export_html_report(html_path)
    
    print("\n" + "="*50)
    print(" PIPELINE EXECUTION SUCCESSFUL ".center(50, "="))
    print("="*50)
    print(f"[*] Annotated video output saved: '{args.output}'")
    print(f"[*] CSV event logs exported   : '{csv_path}'")
    print(f"[*] JSON session summary      : '{json_path}'")
    print(f"[*] Premium HTML Report       : '{html_path}'")
    print("="*50)

if __name__ == "__main__":
    main()
