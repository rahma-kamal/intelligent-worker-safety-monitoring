import os
from ultralytics import YOLO
import yaml
import torch

class WorkerTracker:
    """
    Integrates state-of-the-art YOLOv8 object detection and multi-object tracking (ByteTrack)
    to identify workers in crowded factory environments and maintain stable tracking IDs.
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Select device
        pref_device = self.config.get('device', 'auto')
        if pref_device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = pref_device
            
        model_name = self.config['yolo_model']
        print(f"[*] Loading YOLOv8 pose model '{model_name}' on device '{self.device}'...")
        
        # Load pre-trained model (will automatically download from ultralytics if missing)
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        self.conf_thresh = self.config['confidence_threshold']
        
    def detect_and_track(self, frame):
        """
        Processes a single frame, running detection and updating trackers.
        
        Args:
            frame (np.ndarray): Input image frame (BGR format)
            
        Returns:
            results: Ultralytics Results object containing tracks, keypoints, and boxes
        """
        # Run YOLOv8 built-in ByteTrack multi-object tracker
        # persist=True maintains IDs across frames
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
            tracker="bytetrack.yaml"  # Built-in optimized ByteTrack
        )
        
        # In case tracker fails or returns empty list
        if not results or len(results) == 0:
            return None
            
        return results[0]
