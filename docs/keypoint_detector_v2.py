#!/usr/bin/env python3
"""
ANIMUS // Keypoint Detection System — Lab #6
Assassin's Creed Theme | YOLOv8-Pose | Jetson Orin | JP6

POSE ANOMALY RULES:
  NORMAL  → standing, sitting, walking poses
  ANOMALY → arms raised above shoulders (threat pose)
  ANOMALY → person fallen (low hip keypoints)
  ANOMALY → missing keypoints (occluded/obscured person)
  ANOMALY → >2 persons in frame (crowd)

Streams:
  MJPEG video  → http://localhost:8080/stream
  WebSocket    → ws://localhost:8765

Usage:
    conda activate dev_38
    pip install ultralytics websockets aiohttp --index-url https://pypi.org/simple/
    python3 keypoint_detector.py --camera 0
"""

import argparse
import asyncio
import csv
import json
import math
import os
import time
import random
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("[WARNING] pip install websockets --index-url https://pypi.org/simple/")

try:
    from aiohttp import web
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    print("[WARNING] pip install aiohttp --index-url https://pypi.org/simple/")

# ════════════════════════════════════════════
#   KEYPOINT NAMES (COCO 17)
# ════════════════════════════════════════════
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections for drawing
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

# ════════════════════════════════════════════
#   RULES
# ════════════════════════════════════════════
RULES = {
    "max_persons":          2,      # >2 persons = crowd anomaly
    "arms_raised_thresh":   0.15,   # wrist y < shoulder y - thresh*height = raised
    "fallen_thresh":        0.75,   # hip y > height * thresh = fallen
    "min_keypoints":        8,      # fewer than this = occluded anomaly
    "min_confidence":       0.45,
    "kp_confidence":        0.40,   # min keypoint confidence to use
    "cooldown_seconds":     2.0,
    "img_size":             640,
    # ── Social Distancing ──────────────────────
    "social_distance_px":   150,    # pixel threshold for social distancing
}

OUTPUT_DIR = Path("output")
LOG_FILE   = OUTPUT_DIR / "keypoint_log.csv"
IMG_DIR    = OUTPUT_DIR / "keypoint_images"

# Assassin's Creed BGR palette
C_GOLD   = ( 75, 168, 200)   # gold BGR
C_RED    = (  0,  51, 204)   # red BGR
C_GREEN  = ( 48, 138,  74)   # green BGR
C_WHITE  = (144, 216, 232)   # cream BGR
C_DIM    = ( 32,  46,  58)   # dim BGR
C_PANEL  = (  5,  13,  15)   # dark panel BGR
FONT     = cv2.FONT_HERSHEY_SIMPLEX
FONT_MON = cv2.FONT_HERSHEY_PLAIN

# ════════════════════════════════════════════
#   STATE
# ════════════════════════════════════════════
last_log_time = defaultdict(float)
ws_clients    = set()
person_ids    = {}
latest_frame  = None
stats = {
    "total_frames":    0,
    "total_anomalies": 0,
    "last_anomaly":    "None",
    "last_anomaly_ts": None,
    "persons":         0,
    "start_time":      time.time(),
}

# ════════════════════════════════════════════
#   SETUP
# ════════════════════════════════════════════
def setup_output():
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp","frame","rule","severity","detail","persons_in_frame"
            ])
    print(f"[ANIMUS] Log : {LOG_FILE.resolve()}")
    print(f"[ANIMUS] Imgs: {IMG_DIR.resolve()}")

def make_id():
    return "TGT-" + "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", k=3))

def secs_since(ts):
    return round(time.time()-ts, 1) if ts else None

def pulse(t, speed=3.0):
    return 0.5 + 0.5 * math.sin(t * speed)

# ════════════════════════════════════════════
#   KEYPOINT EXTRACTION
# ════════════════════════════════════════════
def extract_keypoints(kps_data, conf_data, img_w, img_h):
    """
    Returns list of {name, x, y, conf, visible} dicts
    """
    keypoints = []
    for i, name in enumerate(KP_NAMES):
        if i < len(kps_data):
            x = float(kps_data[i][0])
            y = float(kps_data[i][1])
            conf = float(conf_data[i]) if conf_data is not None else 1.0
            keypoints.append({
                "name": name, "x": x, "y": y,
                "conf": conf,
                "visible": conf >= RULES["kp_confidence"],
                "nx": x / img_w,   # normalized
                "ny": y / img_h,
            })
        else:
            keypoints.append({
                "name": name, "x": 0, "y": 0,
                "conf": 0, "visible": False,
                "nx": 0, "ny": 0,
            })
    return keypoints

# ════════════════════════════════════════════
#   POSE ANOMALY RULES
# ════════════════════════════════════════════
def evaluate_pose(keypoints, person_idx, img_h, frame_idx):
    """
    Returns list of violation dicts for one person
    """
    violations = []
    now = time.time()

    visible = [k for k in keypoints if k["visible"]]
    vis_count = len(visible)

    # Get key joint positions (normalized)
    def get_kp(name):
        k = next((k for k in keypoints if k["name"]==name), None)
        return k if (k and k["visible"]) else None

    l_shoulder = get_kp("left_shoulder")
    r_shoulder = get_kp("right_shoulder")
    l_wrist    = get_kp("left_wrist")
    r_wrist    = get_kp("right_wrist")
    l_hip      = get_kp("left_hip")
    r_hip      = get_kp("right_hip")
    nose       = get_kp("nose")

    # ── Rule 1: Arms raised ──────────────────────
    arms_raised = False
    if l_shoulder and l_wrist:
        if l_wrist["ny"] < l_shoulder["ny"] - 0.05:
            arms_raised = True
    if r_shoulder and r_wrist:
        if r_wrist["ny"] < r_shoulder["ny"] - 0.05:
            arms_raised = True
    if arms_raised:
        key = f"arms_raised_{person_idx}"
        if now - last_log_time[key] >= RULES["cooldown_seconds"]:
            violations.append({
                "key": key, "rule": "arms_raised",
                "severity": "critical",
                "detail": f"Person {person_idx+1}: Arms raised above shoulders",
            })
            last_log_time[key] = now

    # ── Rule 2: Person fallen ────────────────────
    if l_hip and r_hip:
        hip_y = (l_hip["ny"] + r_hip["ny"]) / 2
        if hip_y > RULES["fallen_thresh"]:
            key = f"fallen_{person_idx}"
            if now - last_log_time[key] >= RULES["cooldown_seconds"]:
                violations.append({
                    "key": key, "rule": "fallen",
                    "severity": "critical",
                    "detail": f"Person {person_idx+1}: Fallen/lying position detected",
                })
                last_log_time[key] = now

    # ── Rule 3: Missing keypoints (occluded) ─────
    if vis_count < RULES["min_keypoints"]:
        key = f"occluded_{person_idx}"
        if now - last_log_time[key] >= RULES["cooldown_seconds"]:
            violations.append({
                "key": key, "rule": "occluded",
                "severity": "warning",
                "detail": f"Person {person_idx+1}: Only {vis_count}/17 keypoints visible",
            })
            last_log_time[key] = now

    return violations

def evaluate_scene_rules(persons_count, frame_idx):
    """Scene-level rules (not per-person)"""
    violations = []
    now = time.time()

    # Rule 4: Crowd
    if persons_count > RULES["max_persons"]:
        key = "crowd"
        if now - last_log_time[key] >= RULES["cooldown_seconds"]:
            violations.append({
                "key": key, "rule": "crowd",
                "severity": "warning",
                "detail": f"Crowd detected: {persons_count} persons (limit {RULES['max_persons']})",
            })
            last_log_time[key] = now

    return violations

# ════════════════════════════════════════════
#   SOCIAL DISTANCING — Hip Midpoint + Euclidean Distance
# ════════════════════════════════════════════
def get_hip_midpoint(keypoints):
    """
    Extract hip midpoint — same logic as poseNet lab:
      midpoint = (left_hip + right_hip) / 2
    Returns (x, y) or None.
    """
    lh = next((k for k in keypoints if k["name"]=="left_hip"  and k["visible"]), None)
    rh = next((k for k in keypoints if k["name"]=="right_hip" and k["visible"]), None)
    if lh and rh:
        return ((lh["x"]+rh["x"])/2, (lh["y"]+rh["y"])/2)
    if lh: return (lh["x"], lh["y"])
    if rh: return (rh["x"], rh["y"])
    # Fallback to shoulder midpoint if hips not visible
    ls = next((k for k in keypoints if k["name"]=="left_shoulder"  and k["visible"]), None)
    rs = next((k for k in keypoints if k["name"]=="right_shoulder" and k["visible"]), None)
    if ls and rs:
        return ((ls["x"]+rs["x"])/2, (ls["y"]+rs["y"])/2)
    return None

def euclidean_distance(p1, p2):
    """Euclidean pixel distance between two (x,y) points."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def check_social_distancing(all_persons, frame_idx):
    """
    Check all person pairs:
      dist = sqrt((x1-x2)^2 + (y1-y2)^2)
      if dist < threshold -> VIOLATION
    Returns list of violation dicts.
    """
    violations = []
    now = time.time()
    for i in range(len(all_persons)):
        for j in range(i+1, len(all_persons)):
            p1 = all_persons[i].get("hip_mid")
            p2 = all_persons[j].get("hip_mid")
            if p1 is None or p2 is None:
                continue
            dist = euclidean_distance(p1, p2)
            if dist < RULES["social_distance_px"]:
                all_persons[i]["sd_violation"] = True
                all_persons[j]["sd_violation"] = True
                key = f"sd_{i}_{j}"
                if now - last_log_time[key] >= RULES["cooldown_seconds"]:
                    violations.append({
                        "key": key, "rule": "social_distance",
                        "severity": "critical",
                        "detail": f"Persons {i+1}&{j+1} too close: {dist:.0f}px (limit:{RULES['social_distance_px']}px)",
                        "person_a": i, "person_b": j, "distance": dist,
                    })
                    last_log_time[key] = now
    return violations

def draw_distance_line(frame, p1, p2, dist, is_violation, t):
    """Draw line between hip midpoints with distance label."""
    x1,y1 = int(p1[0]),int(p1[1])
    x2,y2 = int(p2[0]),int(p2[1])
    color  = C_RED if is_violation else C_GREEN
    alpha  = 0.6+0.4*pulse(t,4) if is_violation else 0.6
    ov = frame.copy()
    cv2.line(ov,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
    mx,my = (x1+x2)//2,(y1+y2)//2
    label = f"{dist:.0f}px"
    (tw,th),_ = cv2.getTextSize(label,FONT,0.4,1)
    cv2.rectangle(frame,(mx-tw//2-4,my-th-4),(mx+tw//2+4,my+2),color,-1)
    cv2.putText(frame,label,(mx-tw//2,my-1),FONT,0.4,C_PANEL,1,cv2.LINE_AA)

def draw_hip_marker(frame, hip, person_idx, is_violation, t):
    """Draw hip midpoint crosshair marker."""
    x,y   = int(hip[0]),int(hip[1])
    color = C_RED if is_violation else C_GOLD
    ov = frame.copy()
    cv2.circle(ov,(x,y),8,color,-1)
    cv2.addWeighted(ov,0.3*pulse(t,5),frame,1-0.3*pulse(t,5),0,frame)
    cv2.circle(frame,(x,y),8,color,2)
    cv2.circle(frame,(x,y),3,C_WHITE,-1)
    cv2.line(frame,(x-14,y),(x-9,y),color,1)
    cv2.line(frame,(x+9,y),(x+14,y),color,1)
    cv2.line(frame,(x,y-14),(x,y-9),color,1)
    cv2.line(frame,(x,y+9),(x,y+14),color,1)
    cv2.putText(frame,f"HIP-{person_idx+1}",(x+12,y+4),FONT_MON,0.75,color,1,cv2.LINE_AA)

# ════════════════════════════════════════════
#   LOGGING
# ════════════════════════════════════════════
def log_anomaly(frame_idx, v, persons_count, frame=None):
    ts = datetime.now().isoformat(timespec="seconds")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, f"{frame_idx:06d}", v["rule"], v["severity"],
            v["detail"], persons_count
        ])
    print(f"[ANOMALY] {ts} | {v['severity'].upper()} | {v['detail']}")
    stats["total_anomalies"] += 1
    stats["last_anomaly"]    = v["detail"]
    stats["last_anomaly_ts"] = time.time()
    if frame is not None:
        fname = IMG_DIR / f"anomaly_{ts.replace(':','-')}_{v['key']}.jpg"
        cv2.imwrite(str(fname), frame)

# ════════════════════════════════════════════
#   DRAW — Assassin's Creed overlay
# ════════════════════════════════════════════
def draw_panel_bg(frame, x, y, w, h, alpha=0.78):
    ov = frame.copy()
    cv2.rectangle(ov,(x,y),(x+w,y+h),C_PANEL,-1)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(32,46,58),1)

def draw_corner_box(frame, x1, y1, x2, y2, color, length=14, thickness=1):
    pts  = [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]
    dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
    for (cx,cy),(dx,dy) in zip(pts,dirs):
        cv2.line(frame,(cx,cy),(cx+dx*length,cy),color,thickness)
        cv2.line(frame,(cx,cy),(cx,cy+dy*length),color,thickness)

def draw_bar(frame, x, y, w, h, pct, color):
    cv2.rectangle(frame,(x,y),(x+w,y+h),C_DIM,1)
    fill=int(w*max(0.0,min(1.0,pct)))
    if fill>0: cv2.rectangle(frame,(x+1,y+1),(x+fill,y+h-1),color,-1)

def draw_grid(frame):
    fh,fw = frame.shape[:2]
    ov = frame.copy()
    for x in range(0,fw,60): cv2.line(ov,(x,0),(x,fh),(75,168,200),1)
    for y in range(0,fh,60): cv2.line(ov,(0,y),(fw,y),(75,168,200),1)
    cv2.addWeighted(ov,0.03,frame,0.97,0,frame)

def draw_scanline(frame, fi):
    fh,fw = frame.shape[:2]
    y = int((fi*2)%fh)
    ov = frame.copy()
    cv2.rectangle(ov,(0,y),(fw,y+3),C_GOLD,-1)
    cv2.addWeighted(ov,0.04,frame,0.96,0,frame)

def draw_skeleton(frame, keypoints, color):
    """Draw skeleton lines between keypoints"""
    for (a,b) in SKELETON:
        if (keypoints[a]["visible"] and keypoints[b]["visible"]):
            x1,y1 = int(keypoints[a]["x"]), int(keypoints[a]["y"])
            x2,y2 = int(keypoints[b]["x"]), int(keypoints[b]["y"])
            cv2.line(frame,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)

def draw_keypoints(frame, keypoints, color):
    """Draw keypoint circles"""
    for i,kp in enumerate(keypoints):
        if not kp["visible"]: continue
        x,y = int(kp["x"]), int(kp["y"])
        # Face keypoints white, body gold/red
        dot_color = C_WHITE if i < 5 else color
        cv2.circle(frame,(x,y),4,dot_color,-1)
        cv2.circle(frame,(x,y),4,color,1)

def draw_person(frame, keypoints, bbox, tid, is_anomaly, violations, t):
    x1,y1,x2,y2 = bbox
    color = C_RED if is_anomaly else C_GOLD

    # Pulsing fill for anomaly
    if is_anomaly:
        ov = frame.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),C_RED,-1)
        cv2.addWeighted(ov,0.1*pulse(t,5),frame,1-0.1*pulse(t,5),0,frame)

    # Corner box
    draw_corner_box(frame,x1,y1,x2,y2,color,length=16,thickness=2)

    # Skeleton + keypoints
    draw_skeleton(frame,keypoints,color)
    draw_keypoints(frame,keypoints,color)

    # Label
    vis_count = sum(1 for k in keypoints if k["visible"])
    label = f"{'!! ANOMALY' if is_anomaly else 'TARGET'} | {tid} | KP:{vis_count}/17"
    (tw,th),_ = cv2.getTextSize(label,FONT,0.35,1)
    cv2.rectangle(frame,(x1,y1-th-7),(x1+tw+8,y1),color,-1)
    cv2.putText(frame,label,(x1+4,y1-3),FONT,0.35,C_PANEL,1,cv2.LINE_AA)

    # Violation tags
    if violations:
        for i,v in enumerate(violations[:2]):
            cv2.putText(frame,v["detail"][:35],(x1,y2+13+i*12),
                FONT_MON,0.7,C_RED,1,cv2.LINE_AA)

def draw_topbar(frame, fps, fi, has_anomaly):
    fh,fw = frame.shape[:2]
    draw_panel_bg(frame,0,0,fw,24,alpha=0.90)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame,
        f"ANIMUS // KEYPOINT ANALYZER  |  {ts}  |  FRAME:{fi:05d}  |  FPS:{fps:.1f}",
        (8,16),FONT_MON,0.9,C_GOLD,1,cv2.LINE_AA)
    sc = C_RED if has_anomaly else C_GREEN
    label = "!! THREAT DETECTED" if has_anomaly else "SYNCHRONIZED"
    cv2.putText(frame,label,(fw-170,16),FONT_MON,0.9,sc,1,cv2.LINE_AA)
    if int(time.time()*2)%2==0:
        cv2.circle(frame,(fw-180,12),4,sc,-1)

def draw_side_panel(frame, all_keypoints, violations, fi, persons_count, fps):
    fh,fw = frame.shape[:2]
    pw=220; px=fw-pw-2; py=28; ph=fh-32
    draw_panel_bg(frame,px,py,pw,ph,alpha=0.88)
    y=py+14

    # Title
    cv2.putText(frame,"[ ANIMUS TRACKING ]",(px+6,y),FONT_MON,0.9,C_GOLD,1,cv2.LINE_AA); y+=20

    # Sync bar
    sync_pct=min(stats["total_frames"]/100,1.0)
    cv2.putText(frame,"SYNC:",(px+6,y),FONT_MON,0.8,C_DIM,1,cv2.LINE_AA)
    draw_bar(frame,px+45,y-8,pw-55,7,sync_pct,C_GOLD); y+=16

    # Stats
    cv2.line(frame,(px+4,y),(px+pw-4,y),(32,46,58),1); y+=12
    cv2.putText(frame,f"PERSONS  : {persons_count}",(px+6,y),FONT_MON,0.82,C_WHITE,1,cv2.LINE_AA); y+=13
    cv2.putText(frame,f"ANOMALIES: {stats['total_anomalies']}",(px+6,y),FONT_MON,0.82,
        C_RED if stats["total_anomalies"] else C_WHITE,1,cv2.LINE_AA); y+=13
    secs=secs_since(stats["last_anomaly_ts"])
    since=f"{secs}s ago" if secs is not None else "N/A"
    cv2.putText(frame,f"LAST     : {since}",(px+6,y),FONT_MON,0.82,(0,170,255),1,cv2.LINE_AA); y+=16

    # Keypoints for first person
    cv2.line(frame,(px+4,y),(px+pw-4,y),(32,46,58),1); y+=12
    cv2.putText(frame,"[ KEYPOINTS ]",(px+6,y),FONT_MON,0.9,C_GOLD,1,cv2.LINE_AA); y+=13
    if all_keypoints:
        kps = all_keypoints[0]
        for kp in kps[:8]:
            clr = C_GREEN if kp["visible"] else (32,46,58)
            pct = f"{int(kp['conf']*100):3d}%" if kp["conf"]>0 else " ---"
            name = kp["name"].replace("_"," ")[:12]
            cv2.putText(frame,f"{name:<14}{pct}",(px+6,y),FONT_MON,0.72,clr,1,cv2.LINE_AA); y+=11
    else:
        cv2.putText(frame,"No targets in scene",(px+6,y),FONT_MON,0.78,(32,46,58),1,cv2.LINE_AA); y+=11

    # Violations
    y=max(y,py+ph-65)
    cv2.line(frame,(px+4,y),(px+pw-4,y),(32,46,58),1); y+=12
    cv2.putText(frame,"[ VIOLATIONS ]",(px+6,y),FONT_MON,0.9,C_GOLD,1,cv2.LINE_AA); y+=12
    if violations:
        for v in violations[-3:]:
            clr=C_RED if v["severity"]=="critical" else (0,170,255)
            cv2.putText(frame,v["detail"][:23],(px+6,y),FONT_MON,0.72,clr,1,cv2.LINE_AA); y+=11
    else:
        cv2.putText(frame,"No violations",(px+6,y),FONT_MON,0.78,(32,46,58),1,cv2.LINE_AA)

def draw_alert_banner(frame, v, t):
    if pulse(t,speed=5)<0.35: return
    fh,fw=frame.shape[:2]; bh=34; by=fh//2-bh//2
    color=C_RED if v["severity"]=="critical" else (0,170,255)
    ov=frame.copy()
    cv2.rectangle(ov,(0,by),(fw,by+bh),color,-1)
    cv2.addWeighted(ov,0.25,frame,0.75,0,frame)
    cv2.rectangle(frame,(0,by),(fw,by+bh),color,2)
    prefix="!! ANIMUS ALERT" if v["severity"]=="critical" else "!! WARNING"
    msg=f"{prefix}: {v['detail']}"
    (tw,_),_=cv2.getTextSize(msg,FONT,0.55,1)
    cv2.putText(frame,msg,((fw-tw)//2,by+22),FONT,0.55,color,1,cv2.LINE_AA)

def draw_bottombar(frame):
    fh,fw=frame.shape[:2]
    draw_panel_bg(frame,0,fh-18,fw,18,alpha=0.90)
    cv2.putText(frame,
        "NORMAL:STANDING  |  ANOMALY:ARMS RAISED/FALLEN  |  MODEL:YOLOV8N-POSE  |  GPU:ORIN",
        (8,fh-5),FONT_MON,0.75,C_DIM,1,cv2.LINE_AA)

# ════════════════════════════════════════════
#   MJPEG SERVER
# ════════════════════════════════════════════
async def mjpeg_handler(request):
    response = web.StreamResponse(status=200,reason='OK',headers={
        'Content-Type':'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control':'no-cache','Access-Control-Allow-Origin':'*',
    })
    await response.prepare(request)
    while True:
        if latest_frame is not None:
            try:
                await response.write(
                    b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+latest_frame+b'\r\n')
            except: break
        await asyncio.sleep(0.033)
    return response

async def start_mjpeg_server():
    app=web.Application()
    app.router.add_get('/stream',mjpeg_handler)
    runner=web.AppRunner(app)
    await runner.setup()
    site=web.TCPSite(runner,'0.0.0.0',8080)
    await site.start()
    print("[MJPEG] Video stream: http://localhost:8080/stream")

# ════════════════════════════════════════════
#   WEBSOCKET
# ════════════════════════════════════════════
async def ws_handler(websocket,_path=None):
    ws_clients.add(websocket)
    print(f"[WS] Client connected ({len(ws_clients)})")
    try: await websocket.wait_closed()
    finally: ws_clients.discard(websocket)

async def broadcast(payload):
    if not ws_clients: return
    msg=json.dumps(payload)
    await asyncio.gather(*(ws.send(msg) for ws in list(ws_clients)),return_exceptions=True)

# ════════════════════════════════════════════
#   MAIN LOOP
# ════════════════════════════════════════════
async def run_detection(args):
    global latest_frame
    setup_output()

    print(f"\n{'='*55}")
    print("  ANIMUS // KEYPOINT + SOCIAL DISTANCING — ONLINE")
    print(f"  Model    : yolov8n-pose.pt")
    print(f"  Rules    : arms raised, fallen, occluded, crowd")
    print(f"  Distance : {RULES['social_distance_px']}px threshold")
    print(f"  Method   : Hip midpoint Euclidean distance")
    print(f"  Video    : http://localhost:8080/stream")
    print(f"  WebSocket: ws://localhost:8765")
    print(f"{'='*55}\n")

    model=YOLO("yolov8n-pose.pt")
    print("Model loaded OK\n")

    src=int(args.camera) if str(args.camera).isdigit() else args.camera
    cap=cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,30)
    if not cap.isOpened():
        print("ERROR: Cannot open camera."); return

    frame_idx=0; fps=0.0; fps_t=time.time(); fps_frames=0
    alert_timer=0.0; active_viol=None

    while True:
        cap.grab()
        ret,frame=cap.retrieve()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue

        fh,fw=frame.shape[:2]
        t=time.time()

        # ── Inference ─────────────────────────────────────────
        try:
            results=model(frame,conf=RULES["min_confidence"],
                imgsz=RULES["img_size"],verbose=False,device="cuda:0")
        except:
            results=model(frame,conf=RULES["min_confidence"],
                imgsz=RULES["img_size"],verbose=False,device="cpu")

        # ── Parse results ─────────────────────────────────────
        all_persons=[]
        all_keypoints=[]
        all_violations=[]

        if results and results[0].keypoints is not None:
            kps_all  = results[0].keypoints.xy.cpu().numpy()
            kps_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None
            boxes    = results[0].boxes

            for i in range(len(kps_all)):
                kps_xy   = kps_all[i]
                kps_c    = kps_conf[i] if kps_conf is not None else None

                keypoints=extract_keypoints(kps_xy,kps_c,fw,fh)
                all_keypoints.append(keypoints)

                # Bounding box
                if boxes is not None and i < len(boxes):
                    bbox=boxes.xyxy[i].cpu().numpy().astype(int)
                    x1,y1,x2,y2=bbox
                else:
                    xs=[k["x"] for k in keypoints if k["visible"]]
                    ys=[k["y"] for k in keypoints if k["visible"]]
                    x1,y1=int(min(xs))-10,int(min(ys))-10
                    x2,y2=int(max(xs))+10,int(max(ys))+10

                # Track ID
                tid=i
                if tid not in person_ids: person_ids[tid]=make_id()

                # Evaluate pose rules
                viols=evaluate_pose(keypoints,i,fh,frame_idx)
                all_violations.extend(viols)

                # Hip midpoint for social distancing
                hip_mid = get_hip_midpoint(keypoints)

                all_persons.append(dict(
                    keypoints=keypoints,bbox=(x1,y1,x2,y2),
                    tid=person_ids[tid],violations=viols,
                    is_anomaly=len(viols)>0,
                    hip_mid=hip_mid,
                    sd_violation=False,
                ))

        # Scene-level rules
        scene_viols=evaluate_scene_rules(len(all_persons),frame_idx)
        all_violations.extend(scene_viols)

        # ── Social distancing check ───────────────────────────
        sd_violations=check_social_distancing(all_persons,frame_idx)
        all_violations.extend(sd_violations)

        # Mark persons with SD violation as anomaly
        for p in all_persons:
            if p["sd_violation"]:
                p["is_anomaly"]=True

        # Log all anomalies
        for v in all_violations:
            log_anomaly(frame_idx,v,len(all_persons),frame=frame.copy())
            active_viol=v; alert_timer=t+3.5

        # ── Draw AC overlay ───────────────────────────────────
        draw_grid(frame)
        draw_scanline(frame,frame_idx)

        # Draw distance lines between all pairs
        for i in range(len(all_persons)):
            for j in range(i+1,len(all_persons)):
                p1=all_persons[i]["hip_mid"]
                p2=all_persons[j]["hip_mid"]
                if p1 and p2:
                    dist=euclidean_distance(p1,p2)
                    is_v=dist<RULES["social_distance_px"]
                    draw_distance_line(frame,p1,p2,dist,is_v,t)

        for p in all_persons:
            draw_person(frame,p["keypoints"],p["bbox"],
                p["tid"],p["is_anomaly"],p["violations"],t)
            # Draw hip midpoint marker
            if p["hip_mid"]:
                draw_hip_marker(frame,p["hip_mid"],
                    all_persons.index(p),p["sd_violation"],t)

        if t<alert_timer and active_viol:
            draw_alert_banner(frame,active_viol,t)

        n_crit=sum(1 for v in all_violations if v["severity"]=="critical")
        threat_pct=min(len(all_persons)*0.1+n_crit*0.4+len(all_violations)*0.1,1.0)

        draw_topbar(frame,fps,frame_idx,bool(all_violations))
        draw_side_panel(frame,all_keypoints,all_violations,frame_idx,len(all_persons),fps)
        draw_bottombar(frame)

        # MJPEG encode
        ret_enc,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
        if ret_enc: latest_frame=jpeg.tobytes()

        # Stats
        stats["total_frames"]=frame_idx
        stats["persons"]=len(all_persons)

        # WebSocket
        if WS_AVAILABLE:
            kp_data=[]
            if all_keypoints:
                kp_data=[{"name":k["name"],"conf":round(k["conf"],3),"visible":k["visible"]}
                         for k in all_keypoints[0]]
            await broadcast({
                "keypoints":kp_data,
                "violations":[{"rule":v["rule"],"severity":v["severity"],"detail":v["detail"]}
                              for v in all_violations],
                "frame":frame_idx,"fps":round(fps,1),
                "threat_pct":round(threat_pct,3),
                "persons":len(all_persons),
                "stats":{
                    "total_frames":stats["total_frames"],
                    "total_anomalies":stats["total_anomalies"],
                    "last_anomaly":stats["last_anomaly"],
                    "since_anomaly":secs_since(stats["last_anomaly_ts"]),
                    "persons":len(all_persons),
                    "uptime":round(time.time()-stats["start_time"],0),
                }
            })

        fps_frames+=1
        if time.time()-fps_t>=1.0:
            fps=fps_frames/(time.time()-fps_t); fps_frames=0; fps_t=time.time()

        frame_idx+=1
        await asyncio.sleep(0)

    cap.release()
    print(f"\n[ANIMUS] Done. Anomalies: {stats['total_anomalies']}")

# ════════════════════════════════════════════
#   ENTRY POINT
# ════════════════════════════════════════════
async def main(args):
    tasks=[run_detection(args)]
    if WS_AVAILABLE:
        tasks.append(websockets.serve(ws_handler,"0.0.0.0",args.ws_port))
        print(f"[WS] ws://0.0.0.0:{args.ws_port}")
    if HTTP_AVAILABLE:
        tasks.append(start_mjpeg_server())
    await asyncio.gather(*tasks)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="ANIMUS Keypoint + Social Distancing — Lab #6")
    parser.add_argument("--camera",   default="0")
    parser.add_argument("--ws-port",  type=int,  default=8765)
    parser.add_argument("--threshold",type=float,default=0.45)
    parser.add_argument("--distance", type=int,  default=150,
                        help="Social distancing threshold in pixels (default: 150)")
    args=parser.parse_args()
    RULES["min_confidence"]    = args.threshold
    RULES["social_distance_px"]= args.distance
    asyncio.run(main(args))
