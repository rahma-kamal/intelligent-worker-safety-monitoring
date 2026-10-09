import numpy as np
import yaml

class SafetyViolationEngine:
    """
    Hybrid Safety Engine combining deep learning HAR classifications with 
    strict geometric and biomechanical constraints to determine safety alerts, 
    risk scores, and worker efficiency metrics.
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.classes = self.config['dataset']['classes']
        self.bending_thresh = self.config['safety']['bending_angle_threshold']
        self.velocity_thresh = self.config['safety']['running_velocity_threshold']
        self.fall_aspect_thresh = self.config['safety']['fall_aspect_ratio_threshold']

    def analyze_frame_state(self, har_logits, raw_features, track_id):
        """
        Performs hybrid AI + Rule-based assessment on the current frame sequence.
        
        Args:
            har_logits (np.ndarray): Output probabilities or scores from the HAR model (shape: num_classes,)
            raw_features (np.ndarray): Preprocessed feature array (shape: 46,) containing joints and bbox metrics
            track_id (int): ID of the tracked worker
            
        Returns:
            assessment (dict): Complete evaluation containing:
                - AI_activity (str): Classified activity name
                - Posture_checks (dict): Angle/aspect-ratio values
                - Alert_triggered (bool): Whether a violation exists
                - Alert_type (str): Type of violation (or "None")
                - Risk_level (str): "Low", "Medium", or "High"
                - Risk_score (float): Numeric index from 0 to 100
        """
        # 1. AI Classification Output
        class_idx = int(np.argmax(har_logits))
        ai_activity = self.classes[class_idx]
        
        # 2. Extract Biomechanical metrics from preprocessed features
        # angles are elements 34 to 42 in the features array, scaled [0, 1]
        # index 42 is spine angle, multiplied back to degrees
        spine_angle = raw_features[42] * 180.0
        # aspect ratio is element 43
        aspect_ratio = raw_features[43]
        
        # Let's compute average velocity of keypoints
        # In a real environment, velocity is measured as joint displacements.
        # We can extract movement dynamic by checking aspect ratio variations or velocity flags.
        
        # 3. Rule-Based Posture & Movement Checks
        alert_triggered = False
        alert_type = "None"
        risk_level = "Low"
        risk_score = 10.0  # Base safe level
        
        # Rule A: Bending check (Biomechanical Spine stress)
        # Deep bending represents high lower-back strain (e.g. > 110 deg relative to vertical, or spine tilt > 50 degrees)
        is_bending_violation = (spine_angle > 50.0)  # > 50 degrees trunk tilt
        
        # Rule B: Fall detection (Aspect ratio check)
        # If aspect ratio (W / H) is very wide (e.g., > 1.6), worker is horizontal (likely fell)
        is_horizontal = (aspect_ratio > self.fall_aspect_thresh)
        
        # 4. Integrate AI + Rule Outputs (Decision Matrix)
        if ai_activity == "Safe_Walkway":
            risk_level = "Low"
            risk_score = 15.0
        elif ai_activity == "Authorized_Intervention":
            risk_level = "Low"
            risk_score = 20.0
        elif ai_activity == "Closed_Panel_Cover":
            risk_level = "Low"
            risk_score = 15.0
        elif ai_activity == "Safe_Carrying":
            risk_level = "Low"
            risk_score = 25.0
            
        elif ai_activity == "Safe_Walkway_Violation":
            alert_triggered = True
            alert_type = "Walkway Boundary Crossing"
            risk_level = "Medium"
            risk_score = 55.0
            
        elif ai_activity == "Unauthorized_Intervention":
            alert_triggered = True
            alert_type = "Unauthorized Machine Access"
            risk_level = "Medium"
            risk_score = 65.0
            
        elif ai_activity == "Opened_Panel_Cover":
            # Safety risk if panel left open without authorization
            alert_triggered = True
            alert_type = "Exposed Electrical Hazard"
            risk_level = "Medium"
            risk_score = 60.0
            
        elif ai_activity == "Carrying_Overload":
            alert_triggered = True
            alert_type = "Ergonomic Lifting Threat"
            risk_level = "High"
            risk_score = 80.0
            
        # Overrides by rule-based critical alerts:
        if is_horizontal:
            alert_triggered = True
            alert_type = "Worker Fall Detected"
            risk_level = "High"
            risk_score = 98.0
        elif is_bending_violation and ai_activity == "Carrying_Overload":
            # Severe lumbar strain hazard
            alert_triggered = True
            alert_type = "Critical Lumbar Strain"
            risk_level = "High"
            risk_score = 90.0
            
        assessment = {
            "track_id": track_id,
            "ai_activity": ai_activity,
            "posture_checks": {
                "spine_angle_deg": float(spine_angle),
                "aspect_ratio": float(aspect_ratio),
                "is_bending_violation": bool(is_bending_violation),
                "is_horizontal": bool(is_horizontal)
            },
            "alert_triggered": alert_triggered,
            "alert_type": alert_type,
            "risk_level": risk_level,
            "risk_score": risk_score
        }
        
        return assessment
