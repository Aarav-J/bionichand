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
espaddress = "http://100.70.0.210/data"
SERVO_MIN = 0    # degrees
SERVO_MAX = 180   # degrees
SMOOTH_ALPHA = 0.2  # EMA smoothing factor
SAVE_CSV = False   # set True to save readings to simulate output.csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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
    # invert so 0 => open, 1 => closed (you can flip if needed)
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
    # invert so 0 => open, 1 => closed (you can flip if needed)
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
    curl = clamp(1.0 - raw_norm)
    return curl  

def compute_better_thumb_curl(landmarks, handedness=None, use_axis_proj=True, alpha_smooth=0.2, prev_thumb=None): 
    """
    Returns a smoothed curl estimate (0=open .. 1=closed).
    - Angle-based: angle between (TIP->MCP) and (CMC->MCP), normalized.
    - Axis-projection: project TIP->MCP onto hand-x (index_mcp - wrist) and normalize.
    - handedness: 'Left' or 'Right' can be used to flip direction if needed.
    - prev_thumb: previous smoothed value for EMA smoothing.
    """
    tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]

    # vectors in image plane
    v_tip = np.array([tip.x - mcp.x, tip.y - mcp.y], dtype=float)
    v_base = np.array([cmc.x - mcp.x, cmc.y - mcp.y], dtype=float)

    # angle method (robust to scale)
    denom = np.linalg.norm(v_tip) * np.linalg.norm(v_base)
    if denom < 1e-6:
        return None  # not enough signal
    cos = np.clip(np.dot(v_tip, v_base) / denom, -1.0, 1.0)
    angle = math.acos(cos)  # 0..pi

    # Normalize angle to [0..1]. Adjust MAX_ANGLE empirically ( ~60-90 deg )
    MAX_ANGLE = math.radians(80)
    angle_norm = clamp(angle / MAX_ANGLE)

    # axis-projection method (good for primarily horizontal motion)
    hand_x = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y], dtype=float)
    hx_norm = np.linalg.norm(hand_x)
    proj_norm = None
    if hx_norm > 1e-6:
        hand_x /= hx_norm
        proj = np.dot(v_tip, hand_x)  # signed projection
        # normalize projection by an estimate of hand width in x
        hand_scale_x = abs(index_mcp.x - wrist.x) + 1e-6
        # map proj roughly into [0..1] around expected range; tweak if needed
        proj_norm = clamp((proj + hand_scale_x) / (2.0 * hand_scale_x))

    # choose method: prefer axis projection if requested and valid
    if use_axis_proj and proj_norm is not None:
        curl_raw = clamp(1.0 - proj_norm)  # invert so 0=open, 1=closed (flip if mirrored)
    else:
        # angle_norm: larger angle => more curl; tweak inversion if your camera is mirrored
        curl_raw = clamp(angle_norm)

    # optional handedness flip (tweak depending on camera mirroring)
    if handedness is not None and handedness.lower().startswith("right"):
        # if direction is inverted for right hand, flip. Test and remove/adjust as necessary.
        curl_raw = 1.0 - curl_raw

    # EMA smoothing
    if prev_thumb is None:
        smoothed = curl_raw
    else:
        smoothed = alpha_smooth * curl_raw + (1 - alpha_smooth) * prev_thumb

    return smoothed

def compute_thumb_curl_to_palm(landmarks, prev_thumb=None, alpha=SMOOTH_ALPHA, combo_weight=0.6):
    tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    # palm centroid (2D)
    cx = (wrist.x + index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 5.0
    cy = (wrist.y + index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 5.0

    # palm-distance measure
    d_tip_palm = math.hypot(tip.x - cx, tip.y - cy)
    hand_scale = dist(wrist, middle_mcp)  # scale reference
    if hand_scale <= 1e-6:
        return None
    palm_norm = clamp(d_tip_palm / hand_scale)  # larger when tip far from palm
    palm_curl = clamp(1.0 - palm_norm)  # 0=open (far), 1=closed (near)

    # x-projection measure (signed x movement relative to hand width)
    dx = (tip.x - mcp.x)
    hand_scale_x = abs(index_mcp.x - wrist.x) + 1e-6
    proj_norm = clamp((dx + hand_scale_x) / (2.0 * hand_scale_x))  # approx [0..1]
    x_curl = clamp(1.0 - proj_norm)  # invert so moving inward increases curl

    # combine measures (adjust combo_weight to prefer one)
    curl_raw = clamp(combo_weight * x_curl + (1.0 - combo_weight) * palm_curl)

    # EMA smoothing
    if prev_thumb is None:
        smoothed = curl_raw
    else:
        smoothed = alpha * curl_raw + (1.0 - alpha) * prev_thumb

    return smoothed
    
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

def check_threshold(angle, prev_angle, max_delta=10):
    if prev_angle is None:
        return True
    if abs(angle - prev_angle) > max_delta:
        return True  # ignore sudden large changes
    return False
def main():
    cap = cv2.VideoCapture(0)
    prev_angles = [None, None, None, None]  # for index, middle, ring, pinky, 
    csv_rows = []
    

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                time.sleep(0.01)
                continue

            # flip the frame horizontally so processing and display use the selfie view
            image = cv2.flip(image, 1)

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            image.flags.writeable = True
            image_out = image.copy()

            if results.multi_hand_landmarks:
                # use first detected hand
                hand_landmarks = results.multi_hand_landmarks[0].landmark
                curl_index = compute_index_curl(hand_landmarks)
                curl_middle = compute_middle_curl(hand_landmarks)
                curl_ring = compute_ring_curl(hand_landmarks)
                curl_pinky = compute_pinky_curl(hand_landmarks)
                # curl_thumb = compute_thumb_curl_to_palm(hand_landmarks)
                angle_index = compute_angle(curl_index)
                angle_middle = compute_angle(curl_middle)
                angle_ring = compute_angle(curl_ring)
                angle_pinky = compute_angle(curl_pinky)
                # if(angle_index > 100 and angle_middle > 100 and angle_ring > 100 and angle_pinky > 100):
                #     requests.post(espaddress, json={"cmd": "set_value", "value": 0, "led": True})

                # if(check_threshold(angle_index, prev_angles[0])):
                #     prev_angles[0] = angle_index
                #     requests.post(espaddress, json={"finger": "index", "angle": angle_index})
                # if(check_threshold(angle_middle, prev_angles[1])):
                #     prev_angles[1] = angle_middle
                #     requests.post(espaddress, json={"finger": "middle", "angle": angle_middle})
                # if(check_threshold(angle_ring, prev_angles[2])):
                #     prev_angles[2] = angle_ring
                #     requests.post(espaddress, json={"finger": "ring", "angle": angle_ring})
                # if(check_threshold(angle_pinky, prev_angles[3])):
                #     prev_angles[3] = angle_pinky
                #     requests.post(espaddress, json={"finger": "pinky", "angle": angle_pinky})
            
                print(f"[{time.time()-start:5.2f}s] Index servo angle: {angle_index} (curl={curl_index:.3f})")
                print(f"[{time.time()-start:5.2f}s] Middle servo angle: {angle_middle} (curl={curl_middle:.3f})")
                print(f"[{time.time()-start:5.2f}s] Ring servo angle: {angle_ring} (curl={curl_ring:.3f})")
                print(f"[{time.time()-start:5.2f}s] Pinky servo angle: {angle_pinky} (curl={curl_pinky:.3f})")
                # print(f"[{time.time()-start:5.2f}s] Thumb curl: {curl_thumb:.3f}")
                print("-----")
                # overlay on image
                cv2.putText(image_out, f"Index angle: {angle_index} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(image_out, f"Middle angle: {angle_middle} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(image_out, f"Ring angle: {angle_ring} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(image_out, f"Pinky angle: {angle_pinky} deg", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                # cv2.putText(image_out, f"Thumb curl: {curl_thumb:.3f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                

                if SAVE_CSV:
                    csv_rows.append((time.time()-start, curl_index, angle_index))
                else:
                    cv2.putText(image_out, "Hand scale error", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.putText(image_out, "No hand detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # image_out is already in selfie orientation (we flipped earlier), show directly
            cv2.imshow("Servo simulator - press ESC to exit", image_out)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    if SAVE_CSV and csv_rows:
        with open("simulated_servo_output.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "curl", "angle"])
            writer.writerows(csv_rows)
        print("Saved simulated_servo_output.csv")
start = time.time()
prev_angles = [None, None, None, None]
_hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def image_processing(image):
    global _hands, prev_angles, start
    try:
        image = cv2.flip(image, 1)

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = _hands.process(image_rgb)

        image.flags.writeable = True
        image_out = image.copy()

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_out,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )


            first_hand = results.multi_hand_landmarks[0].landmark

            curl_index = compute_index_curl(first_hand)
            curl_middle = compute_middle_curl(first_hand)
            curl_ring = compute_ring_curl(first_hand)
            curl_pinky = compute_pinky_curl(first_hand)

            angle_index = compute_angle(curl_index)
            angle_middle = compute_angle(curl_middle)
            angle_ring = compute_angle(curl_ring)
            angle_pinky = compute_angle(curl_pinky)

            # only post if angle is valid and threshold says it changed
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

            # overlay angles on image
            cv2.putText(image_out, f"Index angle: {angle_index} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(image_out, f"Middle angle: {angle_middle} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(image_out, f"Ring angle: {angle_ring} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(image_out, f"Pinky angle: {angle_pinky} deg", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.putText(image_out, "No hand detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        return image_out
    except Exception:
        traceback.print_exc()
        try:
            return cv2.flip(image, 1)
        except Exception:
            return 255 * np.ones((240, 320, 3), dtype=np.uint8)
                

if __name__ == "__main__":
    main()
