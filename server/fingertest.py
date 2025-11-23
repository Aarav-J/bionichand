import cv2
import math
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Finger landmark indices in MediaPipe Hands
FINGER_LM = {
    "Thumb":  [1, 2, 4],      # CMC, MCP, TIP (thumb is a bit different)
    "Index":  [5, 6, 8],      # MCP, PIP, TIP
    "Middle": [9, 10, 12],    # MCP, PIP, TIP
    "Ring":   [13, 14, 16],   # MCP, PIP, TIP
    "Pinky":  [17, 18, 20],   # MCP, PIP, TIP
}

FINGER_TIP_IDX = {
    "Thumb":  4,
    "Index":  8,
    "Middle": 12,
    "Ring":   16,
    "Pinky":  20,
}

# Smoothing state (one value per finger)
prev_curl = {finger: 0.0 for finger in FINGER_LM.keys()}
SMOOTH_ALPHA = 0.7  # higher = smoother but slower to react


def angle_between(v1, v2):
    """Return angle in degrees between vectors v1 and v2."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cosang = np.dot(v1, v2) / (norm1 * norm2)
    # Clamp for numerical safety
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def compute_finger_curl(landmarks, image_w, image_h):
    """
    Compute curl angles for each finger from MediaPipe landmarks.
    Returns dict: {finger_name: curl_angle_0_to_180}
    """
    curls = {}

    # Convert normalized landmarks to pixel coordinates for easier overlay calculations
    pts = [(lm.x * image_w, lm.y * image_h, lm.z) for lm in landmarks]

    for finger, (a_idx, b_idx, c_idx) in FINGER_LM.items():
        A = pts[a_idx]
        B = pts[b_idx]
        C = pts[c_idx]

        # Work only with x, y (z is small and often noisy, but feel free to include)
        v1 = (A[0] - B[0], A[1] - B[1])  # MCP -> PIP (or equivalent)
        v2 = (C[0] - B[0], C[1] - B[1])  # TIP -> PIP

        joint_angle = angle_between(v1, v2)  # 0..180
        # Convert so that 0 = straight, 180 = fully curled
        curl = 180.0 - joint_angle
        curl = max(0.0, min(180.0, curl))  # clamp

        curls[finger] = curl

    return curls


def smooth_and_quantize_curls(raw_curls):
    """
    Smooth curl values over time and snap to nearest multiple of 10.
    Uses exponential moving average for smoothing.
    """
    display_curls = {}

    for finger, raw_value in raw_curls.items():
        prev_value = prev_curl.get(finger, 0.0)
        smoothed = SMOOTH_ALPHA * prev_value + (1.0 - SMOOTH_ALPHA) * raw_value
        prev_curl[finger] = smoothed

        # Quantize to nearest multiple of 10
        q = int(round(smoothed / 10.0) * 10)
        q = max(0, min(180, q))  # ensure still in range
        display_curls[finger] = q

    return display_curls


def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hands
            results = hands.process(image_rgb)
            h, w, _ = frame.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Compute finger curls (scale-invariant)
                    raw_curls = compute_finger_curl(hand_landmarks.landmark, w, h)
                    display_curls = smooth_and_quantize_curls(raw_curls)

                    # Draw text near each fingertip
                    for finger, angle in display_curls.items():
                        tip_idx = FINGER_TIP_IDX[finger]
                        tip_lm = hand_landmarks.landmark[tip_idx]
                        cx, cy = int(tip_lm.x * w), int(tip_lm.y * h)

                        text = f"{finger}: {angle}°"
                        # Slight upward offset so text isn't directly on the fingertip
                        cv2.putText(
                            frame,
                            text,
                            (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA
                        )

            cv2.imshow('Finger Curl Detection', frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()