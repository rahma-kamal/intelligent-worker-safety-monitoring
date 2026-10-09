import cv2
import numpy as np

# Skeletal connections (COCO Format)
LIMB_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Color Palette (BGR)
COLORS = {
    'safe': (129, 185, 16),      # Sleek Emerald Green
    'warning': (11, 158, 245),    # Golden Amber
    'danger': (68, 68, 239),     # Coral Red
    'skeleton': (240, 240, 240), # Clean white for bones
    'joint': (99, 102, 241)      # High-tech Indigo for joints
}

class PoseVisualizer:
    """
    Renders high-fidelity skeletal structures, bounding boxes, and warning HUD indicators
    on raw video frames for premium dashboard representation.
    """
    def __init__(self):
        pass

    def draw_skeleton(self, frame, keypoints, color=COLORS['joint']):
        """
        Draws COCO joints and connection lines on the frame.
        """
        # Draw limb connections
        for p1, p2 in LIMB_CONNECTIONS:
            kp1 = keypoints[p1]
            kp2 = keypoints[p2]
            
            # Draw line if joints have high confidence
            if kp1[2] > 0.4 and kp2[2] > 0.4:
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0]), int(kp2[1]))
                cv2.line(frame, pt1, pt2, COLORS['skeleton'], 2, cv2.LINE_AA)
                
        # Draw joint dots
        for kp in keypoints:
            x, y, conf = kp
            if conf > 0.4:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
                
        # Draw spine vector (mid-shoulder to mid-hip)
        if (keypoints[5][2] > 0.4 and keypoints[6][2] > 0.4 and 
            keypoints[11][2] > 0.4 and keypoints[12][2] > 0.4):
            mid_shoulder = (
                int((keypoints[5][0] + keypoints[6][0]) / 2),
                int((keypoints[5][1] + keypoints[6][1]) / 2)
            )
            mid_hip = (
                int((keypoints[11][0] + keypoints[12][0]) / 2),
                int((keypoints[11][1] + keypoints[12][1]) / 2)
            )
            cv2.line(frame, mid_shoulder, mid_hip, (220, 100, 100), 2, cv2.LINE_AA)

    def draw_hud_card(self, frame, bbox, track_id, activity, risk_level, risk_score, alert_type):
        """
        Draws a high-tech transparent HUD card overlay above the worker bounding box.
        """
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # Color mapping by risk level
        if risk_level == "Low":
            color = COLORS['safe']
        elif risk_level == "Medium":
            color = COLORS['warning']
        else:
            color = COLORS['danger']
            
        # Draw worker bounding box with thin rounded-style borders
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)
        
        # Draw corner brackets for high-tech HUD look
        l = max(10, int((xmax - xmin) * 0.15))
        # Top-Left corner
        cv2.line(frame, (xmin, ymin), (xmin + l, ymin), color, 4, cv2.LINE_AA)
        cv2.line(frame, (xmin, ymin), (xmin, ymin + l), color, 4, cv2.LINE_AA)
        # Top-Right corner
        cv2.line(frame, (xmax, ymin), (xmax - l, ymin), color, 4, cv2.LINE_AA)
        cv2.line(frame, (xmax, ymin), (xmax, ymin + l), color, 4, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(frame, (xmin, ymax), (xmin + l, ymax), color, 4, cv2.LINE_AA)
        cv2.line(frame, (xmin, ymax), (xmin, ymax - l), color, 4, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (xmax, ymax), (xmax - l, ymax), color, 4, cv2.LINE_AA)
        cv2.line(frame, (xmax, ymax), (xmax, ymax - l), color, 4, cv2.LINE_AA)

        # Build card text
        title_text = f"WORKER #{track_id} | Risk: {risk_score:.0f}"
        activity_text = f"Action: {activity.replace('_', ' ')}"
        
        # Determine height/width of box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        t_size1 = cv2.getTextSize(title_text, font, font_scale, thickness)[0]
        t_size2 = cv2.getTextSize(activity_text, font, font_scale, thickness)[0]
        
        box_w = max(t_size1[0], t_size2[0]) + 20
        box_h = 55
        
        # Draw transparent overlay background above the bounding box
        box_xmin = xmin
        box_ymin = max(0, ymin - box_h - 10)
        box_xmax = box_xmin + box_w
        box_ymax = box_ymin + box_h
        
        # Alpha blending for background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_xmin, box_ymin), (box_xmax, box_ymax), (15, 10, 5), -1)
        # Apply alpha blending
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        
        # Draw accent border line
        cv2.rectangle(frame, (box_xmin, box_ymin), (box_xmax, box_ymax), color, 1, cv2.LINE_AA)
        
        # Render texts
        cv2.putText(frame, title_text, (box_xmin + 10, box_ymin + 20), font, font_scale, color, 1, cv2.LINE_AA)
        cv2.putText(frame, activity_text, (box_xmin + 10, box_ymin + 40), font, font_scale, (240, 240, 240), 1, cv2.LINE_AA)
        
        # Draw critical warning strip if violation is active
        if alert_type != "None":
            alert_text = f"CRITICAL: {alert_type.upper()}"
            t_size3 = cv2.getTextSize(alert_text, font, 0.4, 1)[0]
            
            alert_box_h = 20
            alert_ymin = box_ymax + 2
            alert_ymax = alert_ymin + alert_box_h
            
            overlay_alert = frame.copy()
            cv2.rectangle(overlay_alert, (box_xmin, alert_ymin), (box_xmin + t_size3[0] + 20, alert_ymax), (0, 0, 180), -1)
            cv2.addWeighted(overlay_alert, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (box_xmin, alert_ymin), (box_xmin + t_size3[0] + 20, alert_ymax), COLORS['danger'], 1, cv2.LINE_AA)
            cv2.putText(frame, alert_text, (box_xmin + 10, alert_ymin + 14), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
