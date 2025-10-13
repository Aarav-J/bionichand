import time
import csv
import cv2
import mediapipe as mp
import math
from collections import deque
import requests


# Simple mapping parameters
espaddress = "http://100.70.9.64/data"
SERVO_MIN = 20    # degrees
SERVO_MAX = 160   # degrees
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
    prev_angles = [None, None, None, None]  # for index, middle, ring, pinky
    csv_rows = []
    start = time.time()

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

                angle_index = compute_angle(curl_index)
                angle_middle = compute_angle(curl_middle)
                angle_ring = compute_angle(curl_ring)
                angle_pinky = compute_angle(curl_pinky)
                if(angle_index > 100 and angle_middle > 100 and angle_ring > 100 and angle_pinky > 100):
                    requests.post(espaddress, json={"cmd": "set_value", "value": 0, "led": True})

                if(check_threshold(angle_index, prev_angles[0])):
                    prev_angles[0] = angle_index
                    requests.post(espaddress, json={"finger": "index", "angle": angle_index})
                if(check_threshold(angle_middle, prev_angles[1])):
                    prev_angles[1] = angle_middle
                    requests.post(espaddress, json={"finger": "middle", "angle": angle_middle})
                if(check_threshold(angle_ring, prev_angles[2])):
                    prev_angles[2] = angle_ring
                    requests.post(espaddress, json={"finger": "ring", "angle": angle_ring})
                if(check_threshold(angle_pinky, prev_angles[3])):
                    prev_angles[3] = angle_pinky
                    requests.post(espaddress, json={"finger": "pinky", "angle": angle_pinky})
            
                print(f"[{time.time()-start:5.2f}s] Index servo angle: {angle_index} (curl={curl_index:.3f})")
                print(f"[{time.time()-start:5.2f}s] Middle servo angle: {angle_middle} (curl={curl_middle:.3f})")
                print(f"[{time.time()-start:5.2f}s] Ring servo angle: {angle_ring} (curl={curl_ring:.3f})")
                print(f"[{time.time()-start:5.2f}s] Pinky servo angle: {angle_pinky} (curl={curl_pinky:.3f})")
                print("-----")
                # overlay on image
                cv2.putText(image_out, f"Index angle: {angle_index} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(image_out, f"Middle angle: {angle_middle} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(image_out, f"Ring angle: {angle_ring} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(image_out, f"Pinky angle: {angle_pinky} deg", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                

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

if __name__ == "__main__":
    main()
