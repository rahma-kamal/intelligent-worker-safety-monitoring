import os
import numpy as np
import torch
import yaml

from .tracker import WorkerTracker
from .pose import PoseVisualizer, COLORS
from .safety_engine import SafetyViolationEngine
from data.preprocess import extract_skeletal_features, calculate_velocities
from models import BiLSTMHARModel, TCNHARModel, TransformerHARModel

class WorkerMonitoringPipeline:
    """
    High-level orchestrator connecting YOLO detection/tracking, pose extraction,
    temporal sequence buffers, deep learning activity classification, safety alert checks,
    and visual rendering.
    """
    def __init__(self, model_type="bilstm", checkpoint_path=None, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Select device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('device') != 'cpu' else 'cpu')
        
        # 1. Initialize Components
        self.tracker = WorkerTracker(config_path)
        self.visualizer = PoseVisualizer()
        self.safety_engine = SafetyViolationEngine(config_path)
        
        self.seq_len = self.config['dataset']['sequence_length']
        self.classes = self.config['dataset']['classes']
        self.num_classes = len(self.classes)
        self.num_features = 92
        
        # 2. Sequence buffers per tracked worker
        # Format: {track_id: list of feature arrays of shape (46,)}
        self.track_buffers = {}
        
        # 3. Load active HAR Sequence Model
        self.model_type = model_type.lower()
        self.model = self._initialize_model()
        self.model.to(self.device)
        self.model.eval()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading model checkpoint from '{checkpoint_path}'...")
            self.model.load_checkpoint(checkpoint_path, self.device)
        else:
            print("[!] WARNING: No model checkpoint provided or found. Utilizing unitially-random weights for pipeline execution.")

    def _initialize_model(self):
        """
        Instantiates appropriate PyTorch neural network based on config.
        """
        if self.model_type == "bilstm":
            opts = self.config['models']['bilstm']
            return BiLSTMHARModel(
                input_dim=self.num_features,
                num_classes=self.num_classes,
                hidden_size=opts['hidden_size'],
                num_layers=opts['num_layers'],
                dropout=opts['dropout']
            )
        elif self.model_type == "tcn":
            opts = self.config['models']['tcn']
            return TCNHARModel(
                input_dim=self.num_features,
                num_classes=self.num_classes,
                num_channels=opts['num_channels'],
                kernel_size=opts['kernel_size'],
                dropout=opts['dropout']
            )
        elif self.model_type == "transformer":
            opts = self.config['models']['transformer']
            return TransformerHARModel(
                input_dim=self.num_features,
                num_classes=self.num_classes,
                d_model=opts['d_model'],
                nhead=opts['nhead'],
                num_layers=opts['num_layers'],
                dim_feedforward=opts['dim_feedforward'],
                dropout=opts['dropout']
            )
        else:
            raise ValueError(f"Unknown model architecture type: '{self.model_type}'")

    def process_frame(self, frame, report_generator=None, frame_idx=None, fps=24.0):
        """
        Processes a single frame: extracts keypoints, runs tracking, feeds sequences,
        runs safety validations, and overlays visual graphics on the frame.
        
        Args:
            frame (np.ndarray): raw input video frame (BGR)
            report_generator (SafetyReportGenerator, optional): logs events for reporting
            frame_idx (int, optional): current index of the frame in the video sequence
            fps (float, optional): video frames per second
            
        Returns:
            annotated_frame (np.ndarray): frame with HUD overlays, skeletons, and alert tags
            frame_events (list): safety assessments for each active worker in this frame
        """
        if frame_idx is None:
            if not hasattr(self, 'frame_counter'):
                self.frame_counter = 0
            frame_idx = self.frame_counter
            self.frame_counter += 1
            
        elapsed_seconds = frame_idx / fps
        
        annotated_frame = frame.copy()
        frame_events = []
        
        # 1. Detect and track workers in this frame
        track_results = self.tracker.detect_and_track(frame)
        if track_results is None or track_results.boxes is None:
            return annotated_frame, frame_events
            
        # Get bounding boxes, track IDs and pose keypoints
        boxes = track_results.boxes.xyxy.cpu().numpy()  # bounding boxes
        keypoints = track_results.keypoints.data.cpu().numpy()  # shape: (num_persons, 17, 3)
        
        # YOLO track IDs (can be missing if YOLO fails to assign track IDs in early frames)
        if track_results.boxes.id is not None:
            track_ids = track_results.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(boxes))  # Fallback to local index if no tracks
            
        # 2. Iterate through each detected person
        for idx, track_id in enumerate(track_ids):
            bbox = boxes[idx]
            kps = keypoints[idx]  # shape: (17, 3)
            
            # 2a. Extract normalized features for the current frame
            try:
                current_features = extract_skeletal_features(kps, bbox)
            except Exception as e:
                # Fallback if coordinates are out of bounds or empty
                continue
                
            # 2b. Maintain temporal sliding-window buffer
            if track_id not in self.track_buffers:
                self.track_buffers[track_id] = []
                
            self.track_buffers[track_id].append(current_features)
            
            # Restrict buffer length to sequence length (30 frames)
            if len(self.track_buffers[track_id]) > self.seq_len:
                self.track_buffers[track_id].pop(0)
                
            # 2c. Activity Recognition & Safety Checks
            # Initialize default outputs
            har_logits = np.zeros(self.num_classes)
            har_logits[0] = 1.0  # Default to "Safe_Walkway" if buffer is not full
            
            if len(self.track_buffers[track_id]) == self.seq_len:
                # We have a full temporal sequence: package and run deep learning forward pass
                seq_array = np.array(self.track_buffers[track_id], dtype=np.float32)  # shape: (30, 46)
                
                # Compute velocities (first-order temporal differences) on the fly
                seq_vels = np.zeros_like(seq_array)
                seq_vels[1:] = seq_array[1:] - seq_array[:-1]
                
                # Concatenate spatial features and velocities along the last dimension: shape (30, 92)
                seq_combined = np.concatenate([seq_array, seq_vels], axis=-1)
                
                # Convert to tensor: shape (1, 30, 92) for batch dimension 1
                seq_tensor = torch.tensor(seq_combined, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits_tensor = self.model(seq_tensor)
                    har_logits = torch.softmax(logits_tensor, dim=1).squeeze(0).cpu().numpy()
                    
            # 2d. Run hybrid safety logic
            assessment = self.safety_engine.analyze_frame_state(har_logits, current_features, track_id)
            frame_events.append(assessment)
            
            # 2e. Render premium HUD overlays and skeletal joints
            # Select color based on risk levels
            if assessment["risk_level"] == "Low":
                joint_color = COLORS['joint']
            elif assessment["risk_level"] == "Medium":
                joint_color = COLORS['warning']
            else:
                joint_color = COLORS['danger']
                
            # Draw overlay skeleton
            self.visualizer.draw_skeleton(annotated_frame, kps, color=joint_color)
            
            # Draw HUD card with tracking and alert states
            self.visualizer.draw_hud_card(
                annotated_frame, bbox, track_id, 
                assessment["ai_activity"], assessment["risk_level"], 
                assessment["risk_score"], assessment["alert_type"]
            )
            
            # 2f. Log event to report generator
            if report_generator:
                report_generator.add_record(
                    track_id=track_id,
                    activity=assessment["ai_activity"],
                    alert_triggered=assessment["alert_triggered"],
                    alert_type=assessment["alert_type"],
                    risk_score=assessment["risk_score"],
                    spine_angle=assessment["posture_checks"]["spine_angle_deg"],
                    aspect_ratio=assessment["posture_checks"]["aspect_ratio"],
                    elapsed_seconds=elapsed_seconds
                )
                
        # 3. Clean up inactive trackers (if not detected in current frame, we keep buffers briefly, 
        # but if list gets too large we prune it to conserve memory)
        if len(self.track_buffers) > 50:
            active_ids = set(track_ids)
            inactive_ids = [tid for tid in self.track_buffers if tid not in active_ids]
            # Prune oldest
            for tid in inactive_ids[:10]:
                self.track_buffers.pop(tid, None)
                
        return annotated_frame, frame_events
