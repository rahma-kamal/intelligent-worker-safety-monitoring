import os
import json
import csv
import time
from collections import Counter
import numpy as np

class SafetyReportGenerator:
    """
    Orchestrates post-session reporting and safety analytics.
    Aggregates track events and outputs comprehensive summaries in JSON, CSV, and HTML.
    """
    def __init__(self):
        self.records = []
        self.start_time = time.time()

    def add_record(self, track_id, activity, alert_triggered, alert_type, risk_score, spine_angle, aspect_ratio, elapsed_seconds=None):
        """
        Adds a single frame analytics record for active workers.
        """
        if elapsed_seconds is None:
            elapsed_seconds = int(time.time() - self.start_time)
            
        self.records.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "elapsed_seconds": int(elapsed_seconds),
            "track_id": track_id,
            "activity": activity,
            "alert_triggered": alert_triggered,
            "alert_type": alert_type,
            "risk_score": risk_score,
            "spine_angle": spine_angle,
            "aspect_ratio": aspect_ratio
        })

    def get_discrete_violations(self):
        """
        Processes frame-by-frame records to group contiguous alert frames 
        into discrete temporal violations per worker using a state machine.
        """
        if not self.records:
            return []
            
        violations = []
        active_alerts = {}
        GAP_THRESHOLD_SECS = 2  # Gaps smaller than 2 seconds are merged to filter detector tracking drops
        
        # Sort records by elapsed time to ensure chronological sequence
        sorted_records = sorted(self.records, key=lambda x: x["elapsed_seconds"])
        
        for r in sorted_records:
            tid = r["track_id"]
            alert_triggered = r["alert_triggered"]
            alert_type = r["alert_type"]
            t = r["elapsed_seconds"]
            
            if alert_triggered and alert_type != "None" and alert_type != "":
                if tid in active_alerts:
                    # If it's the same alert type, update the time
                    if active_alerts[tid]["alert_type"] == alert_type:
                        active_alerts[tid]["last_seen_time"] = t
                        active_alerts[tid]["frame_count"] += 1
                    else:
                        # Close previous different alert and start new one
                        closed = active_alerts[tid]
                        dur = closed["last_seen_time"] - closed["start_time"]
                        violations.append({
                            "track_id": int(tid),
                            "alert_type": str(closed["alert_type"]),
                            "start_time_sec": int(closed["start_time"]),
                            "end_time_sec": int(closed["last_seen_time"]),
                            "duration_seconds": int(max(dur, 1)),
                            "frame_count": int(closed["frame_count"])
                        })
                        active_alerts[tid] = {
                            "alert_type": alert_type,
                            "start_time": t,
                            "last_seen_time": t,
                            "frame_count": 1
                        }
                else:
                    # New alert active
                    active_alerts[tid] = {
                        "alert_type": alert_type,
                        "start_time": t,
                        "last_seen_time": t,
                        "frame_count": 1
                    }
            else:
                # No alert in this frame
                if tid in active_alerts:
                    # Check if gap threshold exceeded
                    if t - active_alerts[tid]["last_seen_time"] > GAP_THRESHOLD_SECS:
                        closed = active_alerts[tid]
                        dur = closed["last_seen_time"] - closed["start_time"]
                        violations.append({
                            "track_id": int(tid),
                            "alert_type": str(closed["alert_type"]),
                            "start_time_sec": int(closed["start_time"]),
                            "end_time_sec": int(closed["last_seen_time"]),
                            "duration_seconds": int(max(dur, 1)),
                            "frame_count": int(closed["frame_count"])
                        })
                        del active_alerts[tid]
                        
        # Flush remaining active alerts
        for tid, closed in active_alerts.items():
            dur = closed["last_seen_time"] - closed["start_time"]
            violations.append({
                "track_id": int(tid),
                "alert_type": str(closed["alert_type"]),
                "start_time_sec": int(closed["start_time"]),
                "end_time_sec": int(closed["last_seen_time"]),
                "duration_seconds": int(max(dur, 1)),
                "frame_count": int(closed["frame_count"])
            })
            
        return violations

    def get_summary_statistics(self):
        """
        Aggregates session records to calculate global and per-worker indexes, 
        plus discrete safety violation events.
        """
        if not self.records:
            return {
                "session_date": time.strftime("%Y-%m-%d", time.localtime()),
                "total_frames_processed": 0,
                "session_duration_sec": 0,
                "productivity_index": 100.0,
                "safety_index": 100.0,
                "average_risk_score": 0.0,
                "peak_risk_score": 0.0,
                "workers": {},
                "discrete_violations": []
            }
            
        # 1. Group records by track_id
        records_by_worker = {}
        for r in self.records:
            tid = r["track_id"]
            if tid not in records_by_worker:
                records_by_worker[tid] = []
            records_by_worker[tid].append(r)
            
        # 2. Extract discrete violations
        discrete_violations = self.get_discrete_violations()
        
        # 3. Calculate per-worker statistics
        workers_summary = {}
        productive_states = ["Safe_Walkway", "Authorized_Intervention", "Closed_Panel_Cover", "Safe_Carrying"]
        
        for tid, worker_records in records_by_worker.items():
            w_total = len(worker_records)
            w_activities = [wr["activity"] for wr in worker_records]
            w_risk_scores = [wr["risk_score"] for wr in worker_records]
            
            # Count productive frames for this worker
            w_activity_counts = Counter(w_activities)
            w_productive_count = sum(w_activity_counts[state] for state in productive_states)
            w_prod_index = (w_productive_count / w_total) * 100 if w_total > 0 else 100.0
            
            # Count safe frames (< 40 risk score) for this worker
            w_safe_count = sum(1 for wr in worker_records if wr["risk_score"] < 40.0)
            w_safe_index = (w_safe_count / w_total) * 100 if w_total > 0 else 100.0
            
            # Worker-specific discrete violations
            w_violations = [dv for dv in discrete_violations if dv["track_id"] == tid]
            
            workers_summary[str(tid)] = {
                "track_id": int(tid),
                "total_frames": int(w_total),
                "productivity_index": round(float(w_prod_index), 2),
                "safety_index": round(float(w_safe_index), 2),
                "average_risk_score": round(float(np.mean(w_risk_scores)), 2) if w_risk_scores else 0.0,
                "peak_risk_score": float(np.max(w_risk_scores)) if w_risk_scores else 0.0,
                "violations_count": int(len(w_violations)),
                "activity_distribution": {str(act): int(count) for act, count in w_activity_counts.items()}
            }
            
        # 4. Calculate Global Metrics
        total_records = len(self.records)
        global_risk_scores = [r["risk_score"] for r in self.records]
        
        # Average of worker indexes
        global_prod = np.mean([w["productivity_index"] for w in workers_summary.values()]) if workers_summary else 100.0
        global_safe = np.mean([w["safety_index"] for w in workers_summary.values()]) if workers_summary else 100.0
        
        session_duration = max(r["elapsed_seconds"] for r in self.records) if self.records else 0
        
        summary = {
            "session_date": time.strftime("%Y-%m-%d", time.localtime()),
            "total_frames_processed": int(total_records),
            "session_duration_sec": int(session_duration),
            "productivity_index": round(float(global_prod), 2),
            "safety_index": round(float(global_safe), 2),
            "average_risk_score": round(float(np.mean(global_risk_scores)), 2) if global_risk_scores else 0.0,
            "peak_risk_score": float(np.max(global_risk_scores)) if global_risk_scores else 0.0,
            "safety_violations_count": int(len(discrete_violations)),
            "workers": workers_summary,
            "discrete_violations": discrete_violations
        }
        
        return summary

    def export_csv(self, filepath):
        """
        Exports flat log records to a CSV file.
        """
        if not self.records:
            return
            
        keys = self.records[0].keys()
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.records)
        print(f"[SUCCESS] CSV logs exported to '{filepath}'")

    def export_json(self, filepath):
        """
        Exports the compiled analytics summary to a JSON file.
        """
        summary = self.get_summary_statistics()
        if not summary:
            return
            
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"[SUCCESS] JSON summary analytics exported to '{filepath}'")

    def export_html_report(self, filepath):
        """
        Generates a premium, publication-quality HTML report for occupational safety teams.
        Includes glassmorphism, responsive tables, individual worker KPI cards,
        worker activity breakdowns, and a discrete temporal violation timeline.
        """
        summary = self.get_summary_statistics()
        if not summary:
            return
            
        # 1. Compile per-worker HTML sections
        worker_html_blocks = []
        for wid, wdata in sorted(summary["workers"].items(), key=lambda x: int(x[0])):
            # Activity breakdown table rows for this worker
            w_total_frames = wdata["total_frames"]
            act_rows = ""
            for act, count in wdata["activity_distribution"].items():
                pct = round((count / w_total_frames) * 100, 1) if w_total_frames > 0 else 0.0
                act_rows += f"<tr><td>{act.replace('_', ' ')}</td><td>{count}</td><td>{pct}%</td></tr>"
                
            # Style classification color for safety rating
            w_rating_class = "success-text"
            if wdata["safety_index"] < 90.0:
                w_rating_class = "warning-text"
            if wdata["safety_index"] < 70.0:
                w_rating_class = "danger-text"
                
            block = f"""
            <div class="worker-card">
                <div class="worker-header">
                    <h2>Worker #{wid} (Track ID: {wid})</h2>
                    <span class="{w_rating_class}" style="font-size: 1.1rem; font-weight:700;">
                        Safety Rating: {wdata["safety_index"]}%
                    </span>
                </div>
                
                <div class="worker-kpi-row">
                    <div class="w-kpi-box">
                        <div class="val" style="color: var(--primary);">{wdata["productivity_index"]}%</div>
                        <div class="lbl">Productivity Index</div>
                    </div>
                    <div class="w-kpi-box">
                        <div class="val" style="color: var(--warning);">{wdata["average_risk_score"]}</div>
                        <div class="lbl">Avg Risk Score</div>
                    </div>
                    <div class="w-kpi-box">
                        <div class="val" style="color: var(--danger);">{wdata["violations_count"]}</div>
                        <div class="lbl">Violations Count</div>
                    </div>
                </div>

                <div class="worker-sections">
                    <div class="worker-table-col">
                        <div class="subsection-title">Activity Breakdown</div>
                        <table>
                            <thead>
                                <tr>
                                    <th>Activity Class</th>
                                    <th>Frames</th>
                                    <th>Share</th>
                                </tr>
                            </thead>
                            <tbody>
                                {act_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            """
            worker_html_blocks.append(block)
            
        workers_html = "\n".join(worker_html_blocks)
        
        # 2. Compile discrete violations timeline
        violation_rows = ""
        if summary["discrete_violations"]:
            for idx, v in enumerate(summary["discrete_violations"]):
                # Convert seconds to timestamp format MM:SS
                start_m, start_s = divmod(v["start_time_sec"], 60)
                end_m, end_s = divmod(v["end_time_sec"], 60)
                time_range = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"
                
                violation_rows += f"""
                <tr>
                    <td>Incident #{idx+1}</td>
                    <td><span class="worker-tag">Worker #{v["track_id"]}</span></td>
                    <td class="danger-text">{v["alert_type"]}</td>
                    <td>{time_range}</td>
                    <td><strong>{v["duration_seconds"]}s</strong> ({v["frame_count"]} frames)</td>
                </tr>
                """
        else:
            violation_rows = "<tr><td colspan='5' class='success-text' style='text-align:center;'>No safety violations detected in this session.</td></tr>"

        # Determine overall global rating
        overall_rating = "Excellent"
        rating_class = "success-text"
        if summary["safety_index"] < 90.0:
            overall_rating = "Concern"
            rating_class = "warning-text"
        if summary["safety_index"] < 70.0:
            overall_rating = "Hazardous"
            rating_class = "danger-text"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Factory Worker Safety & Productivity Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        :root {{
            --bg-color: #0d0f14;
            --card-bg: rgba(22, 28, 45, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #e2e8f0;
            --text-muted: #94a3b8;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.2);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}

        .report-container {{
            max-width: 950px;
            width: 100%;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            backdrop-filter: blur(16px);
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }}

        .header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 30px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header h1 {{
            font-size: 2rem;
            margin: 0;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header .meta {{
            text-align: right;
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}

        .kpi-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }}

        .kpi-card:hover {{
            border-color: var(--primary);
            box-shadow: 0 0 15px var(--primary-glow);
        }}

        .kpi-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}

        .kpi-card .label {{
            color: var(--text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .section-title {{
            font-size: 1.4rem;
            font-weight: 600;
            margin: 40px 0 20px 0;
            color: #a5b4fc;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section-title::after {{
            content: '';
            flex-grow: 1;
            height: 1px;
            background: var(--border-color);
        }}

        .subsection-title {{
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-color);
        }}

        .worker-card {{
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
        }}

        .worker-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}

        .worker-header h2 {{
            font-size: 1.25rem;
            margin: 0;
            color: var(--text-color);
        }}

        .worker-kpi-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}

        .w-kpi-box {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }}

        .w-kpi-box .val {{
            font-size: 1.4rem;
            font-weight: 700;
        }}

        .w-kpi-box .lbl {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 10px;
        }}

        th {{
            text-align: left;
            padding: 12px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            font-size: 0.9rem;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            font-size: 0.95rem;
        }}

        .worker-tag {{
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid var(--primary);
            color: #a5b4fc;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .success-text {{ color: var(--success); font-weight: 600; }}
        .warning-text {{ color: var(--warning); font-weight: 600; }}
        .danger-text {{ color: var(--danger); font-weight: 600; }}

        .notes-card {{
            background: rgba(99, 102, 241, 0.04);
            border-left: 4px solid var(--primary);
            border-radius: 0 12px 12px 0;
            padding: 20px;
            margin-bottom: 30px;
        }}

        .notes-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.05rem;
            color: #a5b4fc;
        }}

        .notes-card p {{
            margin: 0;
            color: var(--text-muted);
            font-size: 0.95rem;
            line-height: 1.6;
        }}

        .footer {{
            text-align: center;
            border-top: 1px solid var(--border-color);
            padding-top: 30px;
            margin-top: 40px;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>

<div class="report-container">
    <div class="header">
        <div>
            <h1>Factory Safety Analytics</h1>
            <div style="color: var(--text-muted); margin-top: 5px;">Post-Session Operations Evaluation</div>
        </div>
        <div class="meta">
            <div><strong>Date:</strong> {summary["session_date"]}</div>
            <div><strong>Duration:</strong> {summary["session_duration_sec"]}s</div>
            <div><strong>Status:</strong> <span class="{rating_class}">{overall_rating}</span></div>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="value" style="color: var(--primary);">{summary["productivity_index"]}%</div>
            <div class="label">Global Productivity</div>
        </div>
        <div class="kpi-card">
            <div class="value" style="color: var(--success);">{summary["safety_index"]}%</div>
            <div class="label">Global Safety Index</div>
        </div>
        <div class="kpi-card">
            <div class="value" style="color: var(--warning);">{summary["average_risk_score"]}</div>
            <div class="label">Average Risk</div>
        </div>
        <div class="kpi-card">
            <div class="value" style="color: var(--danger);">{summary["safety_violations_count"]}</div>
            <div class="label">Discrete Incidents</div>
        </div>
    </div>

    <div class="notes-card">
        <h3>Executive Safety Summary</h3>
        <p>
            During this session, worker tracking and temporal Human Activity Recognition (HAR) models assessed movement dynamics.
            The worker pool demonstrated an overall safety index of <strong>{summary["safety_index"]}%</strong> with a productivity index of 
            <strong>{summary["productivity_index"]}%</strong>. A total of <strong>{summary["safety_violations_count"]}</strong> discrete hazard incidents were logged. 
            Detailed individual metrics and timeline logs are provided below.
        </p>
    </div>

    <div class="section-title">Individual Worker Directory</div>
    {workers_html}

    <div class="section-title">Safety Incident Timeline</div>
    <table>
        <thead>
            <tr>
                <th>Incident ID</th>
                <th>Worker ID</th>
                <th>Violation Type</th>
                <th>Video Timeline</th>
                <th>Duration (s)</th>
            </tr>
        </thead>
        <tbody>
            {violation_rows}
        </tbody>
    </table>

    <div class="section-title">AI System Recommendations</div>
    <div style="font-size:0.95rem; line-height: 1.7; color: var(--text-muted);">
        <ul style="padding-left: 20px; margin: 0;">
            <li style="margin-bottom: 10px;"><strong style="color:var(--text-color);">Ergonomic Lifting:</strong> Overload carry alerts suggest back angle deviations. Re-train workers on standard squat-lifting methodologies.</li>
            <li style="margin-bottom: 10px;"><strong style="color:var(--text-color);">Boundary Violations:</strong> Ensure that restricted walkway corridors are visibly demarcated to avoid accidental Safe Walkway crossings.</li>
            <li><strong style="color:var(--text-color);">Panel Maintenance:</strong> Keep panels closed. Standard operating procedures must demand closing electrical gates directly after intervention processes.</li>
        </ul>
    </div>

    <div class="footer">
        Generated by Antigravity HAR Platform | AI Operations Auditing Project
    </div>
</div>

</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[SUCCESS] Premium HTML analytics report exported to '{filepath}'")
