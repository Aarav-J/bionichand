import time
import csv
import cv2
import mediapipe as mp
import math
import numpy as np
import requests
import threading
import traceback


# Simple mapping parameters

espaddress = "http://100.70.3.129/data"
SERVO_MIN = 20    # degrees
SERVO_MAX = 160   # degrees
SMOOTH_ALPHA = 0.2  # EMA smoothing factor
SAVE_CSV = False   # set True to save readings to simulate output.csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
FINGER_LM = {
    "Index": (
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
    ),
    "Middle": (
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    ),
    "Ring": (
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
    ),
    "Pinky": (
        mp_hands.HandLandmark.PINKY_MCP,
        mp_hands.HandLandmark.PINKY_PIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ),
}

# For smoothing curls (in degrees) per finger
prev_curl_deg = {name: None for name in FINGER_LM.keys()}


def angle_between(v1, v2):
    """Angle in degrees between 2 vectors v1, v2."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = max(-1.0, min(1.0, cosang))  # numerical safety
    return math.degrees(math.acos(cosang))


def compute_finger_curls_deg(landmarks):
    """
    Compute curl for each finger in degrees (0 = straight, 180 = fully curled).
    Uses joint angle: curl = 180 - angle(MCP->PIP, TIP->PIP).
    Landmarks are normalized; angles are scale-invariant.
    Returns dict: { "Index": deg, "Middle": deg, ... }
    """
    curls = {}

    for name, (a_idx, b_idx, c_idx) in FINGER_LM.items():
        A = landmarks[a_idx]
        B = landmarks[b_idx]
        C = landmarks[c_idx]

        v1 = (A.x - B.x, A.y - B.y, A.z - B.z)
        v2 = (C.x - B.x, C.y - B.y, C.z - B.z)

        joint_angle = angle_between(v1, v2)     # 0..180, 180 = straight
        curl = 180.0 - joint_angle              # 0 = straight, 180 = curled
        curl = clamp(curl, 0.0, 180.0)
        curls[name] = curl

    return curls


def smooth_and_quantize_curls(raw_curls):
    """
    Smooth curl degrees using EMA, then snap to nearest 10 degrees.
    Returns dict: { "Index": int_deg, ... } where int_deg ∈ {0, 10, ..., 180}
    """
    global prev_curl_deg
    quantized = {}

    for name, raw_val in raw_curls.items():
        prev = prev_curl_deg.get(name)
        if prev is None:
            smoothed = raw_val
        else:
            # EMA: more weight on previous value for stability
            smoothed = SMOOTH_ALPHA * raw_val + (1.0 - SMOOTH_ALPHA) * prev

        prev_curl_deg[name] = smoothed

        q = int(round(smoothed / 10.0) * 10)
        q = max(0, min(180, q))
        quantized[name] = q

    return quantized
# Calibration storage and helpers
calib_temp_open = {}
calib_temp_closed = {}
calib_data = {}          # e.g. {'index': (open_avg, closed_avg), ...}
CALIB_FRAMES = 40
calib_collecting = None  # None | 'open' | 'closed'
calib_frames_left = 0
open_avgs_temp = None
FINGER_KEYS = ["index", "middle", "ring", "pinky", "thumb"]

def _init_calib_temps():
    for k in FINGER_KEYS:
        calib_temp_open[k] = []
        calib_temp_closed[k] = []

_init_calib_temps()

def _store_calib_sample_from_landmarks(landmarks, kind):
    """Compute raw_norms for each finger (same as compute_* raw_norm) and store in temp lists."""
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    # index
    try:
        tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        d_raw = dist(tip, mcp)
        hand_scale = dist(wrist, middle_mcp)
        raw_index = None if hand_scale <= 1e-6 else clamp(d_raw / hand_scale)
    except Exception:
        raw_index = None

    # middle
    try:
        tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        d_raw = dist(tip, mcp)
        hand_scale = dist(wrist, index_mcp)
        raw_middle = None if hand_scale <= 1e-6 else clamp(d_raw / hand_scale)
    except Exception:
        raw_middle = None

    # ring
    try:
        tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
        d_raw = dist(tip, mcp)
        hand_scale = dist(wrist, pinky_mcp)
        raw_ring = None if hand_scale <= 1e-6 else clamp(d_raw / hand_scale)
    except Exception:
        raw_ring = None

    # pinky
    try:
        tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
        mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
        ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
        d_raw = dist(tip, mcp)
        hand_scale = dist(wrist, ring_mcp)
        raw_pinky = None if hand_scale <= 1e-6 else clamp(d_raw / hand_scale)
    except Exception:
        raw_pinky = None

    # thumb (use x-projection raw_norm as current implementation)
    try:
        tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
        index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        dx = abs(tip.x - mcp.x)
        hand_scale_x = abs(wrist.x - index_mcp.x)
        raw_thumb = None if hand_scale_x <= 1e-6 else clamp(dx / hand_scale_x)
    except Exception:
        raw_thumb = None

    samples = {
        "index": raw_index,
        "middle": raw_middle,
        "ring": raw_ring,
        "pinky": raw_pinky,
        "thumb": raw_thumb,
    }
    for k, v in samples.items():
        if v is None:
            continue
        if kind == "open":
            calib_temp_open[k].append(v)
        else:
            calib_temp_closed[k].append(v)

def _finalize_calib(kind):
    if kind == 'open':
        open_avgs = {k: (sum(calib_temp_open[k]) / len(calib_temp_open[k]) if calib_temp_open[k] else None) for k in FINGER_KEYS}
        return open_avgs
    else:
        closed_avgs = {k: (sum(calib_temp_closed[k]) / len(calib_temp_closed[k]) if calib_temp_closed[k] else None) for k in FINGER_KEYS}
        return closed_avgs

def apply_calibration(finger, raw_norm):
    """Map raw_norm to calibrated curl using stored open/closed averages."""
    if finger not in calib_data or raw_norm is None:
        return None
    open_avg, closed_avg = calib_data[finger]
    if open_avg is None or closed_avg is None:
        return None
    denom = open_avg - closed_avg
    if abs(denom) < 1e-6:
        return None
    curl = clamp((open_avg - raw_norm) / denom)
    return curl
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))

def map_norm_to_servo(norm):
    # norm: 0 (finger far from mcp/open) .. 1 (finger close/curl)
    angle = SERVO_MIN + (SERVO_MAX - SERVO_MIN) * norm
    return int(clamp(angle, SERVO_MIN, SERVO_MAX))

def _post_async(url, json_payload, timeout=0.8):
    """Fire-and-forget POST with short timeout so network problems can't block processing."""
    def _job():
        try:
            requests.post(url, json=json_payload, timeout=timeout)
        except Exception:
            # swallow errors to avoid killing the processing loop
            pass
    threading.Thread(target=_job, daemon=True).start()

def compute_index_curl(landmarks):
    # landmarks are normalized
    tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    d_raw = dist(tip, mcp)
    hand_scale = dist(wrist, middle_mcp)
    if hand_scale <= 0:
        return None
    raw_norm = clamp(d_raw / hand_scale)  # larger when finger extended
    # try calibration first
    calibrated = apply_calibration("index", raw_norm)
    if calibrated is not None:
        return calibrated
    # fallback: invert so 0 => open, 1 => closed (you can flip if needed)
    curl = clamp(1.0 - raw_norm)
    return curl

def compute_middle_curl(landmarks): 
    tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    d_raw = dist(tip, mcp)
    hand_scale = dist(wrist, index_mcp)
    if hand_scale <= 0:
        return None
    raw_norm = clamp(d_raw / hand_scale)  # larger when finger extended
    calibrated = apply_calibration("middle", raw_norm)
    if calibrated is not None:
        return calibrated
    curl = clamp(1.0 - raw_norm)
    return curl

def compute_ring_curl(landmarks):
    tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    d_raw = dist(tip, mcp)
    hand_scale = dist(wrist, pinky_mcp)
    if hand_scale <= 0:
        return None
    raw_norm = clamp(d_raw / hand_scale)  
    calibrated = apply_calibration("ring", raw_norm)
    if calibrated is not None:
        return calibrated
    curl = clamp(1.0 - raw_norm)
    return curl

def compute_pinky_curl(landmarks):
    tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]

    d_raw = dist(tip, mcp)
    hand_scale = dist(wrist, ring_mcp)
    if hand_scale <= 0:
        return None
    raw_norm = clamp(d_raw / hand_scale)  
    calibrated = apply_calibration("pinky", raw_norm)
    if calibrated is not None:
        return calibrated
    curl = clamp(1.0 - raw_norm)
    return curl
def compute_thumb_curl(landmarks): 
    tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    dx = abs(tip.x-mcp.x)
    hand_scale_x = abs(wrist.x - index_mcp.x)
    if hand_scale_x <= 1e-6: 
        return None
    # d_raw = dist(tip, mcp)
    # hand_scale = dist(wrist, index_mcp)
    # if hand_scale <= 0:
    #     return None
    raw_norm = clamp(dx / hand_scale_x)
    calibrated = apply_calibration("thumb", raw_norm)
    if calibrated is not None:
        return calibrated
    curl = clamp(1.0 - raw_norm)
    return curl

def compute_angle(curl): 
    prev_angle = None
    if curl is not None: 
        target_angle = map_norm_to_servo(curl)
        if prev_angle is None:
            angle = target_angle
        else:
            angle = int(SMOOTH_ALPHA * target_angle + (1 - SMOOTH_ALPHA) * prev_angle)
        prev_angle = angle
        return angle

def check_threshold(angle, prev_angle, max_delta=15):
    if prev_angle is None:
        return True
    if abs(angle - prev_angle) > max_delta:
        return True  # ignore sudden large changes
    return False
def main():
    global start
    cap = cv2.VideoCapture(0)
    csv_rows = []

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                time.sleep(0.01)
                continue

            # Flip for selfie view
            image = cv2.flip(image, 1)

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            image.flags.writeable = True
            image_out = image.copy()
            h, w, _ = image_out.shape

            if results.multi_hand_landmarks:
                # Just use first detected hand
                first_hand = results.multi_hand_landmarks[0].landmark

                # Angle-based curl (scale-invariant)
                raw_curls_deg = compute_finger_curls_deg(first_hand)
                # Smooth + snap to 10°
                angles_deg = smooth_and_quantize_curls(raw_curls_deg)

                angle_index = angles_deg["Index"]
                angle_middle = angles_deg["Middle"]
                angle_ring = angles_deg["Ring"]
                angle_pinky = angles_deg["Pinky"]

                curl_index = raw_curls_deg["Index"]
                curl_middle = raw_curls_deg["Middle"]
                curl_ring = raw_curls_deg["Ring"]
                curl_pinky = raw_curls_deg["Pinky"]

                # Debug prints
                t = time.time() - start
                print(f"[{t:5.2f}s] Index  angle: {angle_index:3d} (curl_raw={curl_index:6.2f})")
                print(f"[{t:5.2f}s] Middle angle: {angle_middle:3d} (curl_raw={curl_middle:6.2f})")
                print(f"[{t:5.2f}s] Ring   angle: {angle_ring:3d} (curl_raw={curl_ring:6.2f})")
                print(f"[{t:5.2f}s] Pinky  angle: {angle_pinky:3d} (curl_raw={curl_pinky:6.2f})")
                print("-----")

                # Overlay text (angles are already multiples of 10)
                cv2.putText(image_out, f"Index:  {angle_index} deg",  (10,  40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image_out, f"Middle: {angle_middle} deg", (10,  90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image_out, f"Ring:   {angle_ring} deg",   (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image_out, f"Pinky:  {angle_pinky} deg",  (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if SAVE_CSV:
                    csv_rows.append((t, curl_index, angle_index))
            else:
                cv2.putText(image_out, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Servo simulator - press ESC to exit", image_out)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    if SAVE_CSV and csv_rows:
        with open("simulated_servo_output.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "curl_raw_deg", "angle_quantized_deg"])
            writer.writerows(csv_rows)
        print("Saved simulated_servo_output.csv")

start = time.time()
prev_angles = [None, None, None, None]
_hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def image_processing(image):
    global _hands, prev_angles, start
    try:
        # Selfie flip
        image = cv2.flip(image, 1)

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = _hands.process(image_rgb)

        image.flags.writeable = True
        image_out = image.copy()
        h, w, _ = image_out.shape

        if results.multi_hand_landmarks:
            # Draw all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_out,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )

            # Use first hand for servo control
            first_hand = results.multi_hand_landmarks[0].landmark

            # Angle-based curls
            raw_curls_deg = compute_finger_curls_deg(first_hand)
            angles_deg = smooth_and_quantize_curls(raw_curls_deg)

            angle_index = angles_deg["Index"]
            angle_middle = angles_deg["Middle"]
            angle_ring = angles_deg["Ring"]
            angle_pinky = angles_deg["Pinky"]

            # POST to ESP only if changed enough
            if angle_index is not None and check_threshold(angle_index, prev_angles[0]):
                prev_angles[0] = angle_index
                _post_async(espaddress, {"finger": "index", "angle": angle_index})

            if angle_middle is not None and check_threshold(angle_middle, prev_angles[1]):
                prev_angles[1] = angle_middle
                _post_async(espaddress, {"finger": "middle", "angle": angle_middle})

            if angle_ring is not None and check_threshold(angle_ring, prev_angles[2]):
                prev_angles[2] = angle_ring
                _post_async(espaddress, {"finger": "ring", "angle": angle_ring})

            if angle_pinky is not None and check_threshold(angle_pinky, prev_angles[3]):
                prev_angles[3] = angle_pinky
                _post_async(espaddress, {"finger": "pinky", "angle": angle_pinky})

            # Overlay angles (already multiples of 10)
            cv2.putText(image_out, f"Index:  {angle_index} deg",  (10,  40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image_out, f"Middle: {angle_middle} deg", (10,  90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image_out, f"Ring:   {angle_ring} deg",   (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image_out, f"Pinky:  {angle_pinky} deg",  (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(image_out, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return image_out

    except Exception:
        traceback.print_exc()
        try:
            return cv2.flip(image, 1)
        except Exception:
            return 255 * np.ones((240, 320, 3), dtype=np.uint8)
if __name__ == "__main__":
    main()
