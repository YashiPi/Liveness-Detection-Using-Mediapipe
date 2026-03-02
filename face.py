import cv2
import mediapipe as mp
import numpy as np
import random
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- CONFIG ----------------
MODEL_PATH = "face_landmarker.task"

CHALLENGES_POOL = ["blink", "smile", "mouth", "turn_left", "turn_right"]
TOTAL_STEPS = 3
STEP_TIME_LIMIT = 5  # seconds

EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 2

blink_counter = 0
blink_state = "open"
blink_total = 0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

BLINK_THRESHOLD = 0.6
SMILE_THRESHOLD = 0.7
MOUTH_THRESHOLD = 0.6
HEAD_TURN_THRESHOLD = 0.03
MOTION_THRESHOLD = 2.0

# ---------------- LOAD MODEL ----------------
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)


# EAR function
import math

def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_ear(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# ---------------- STATE ----------------
challenge_sequence = random.sample(CHALLENGES_POOL, TOTAL_STEPS)
current_step = 0
step_start_time = time.time()
verified = False
failed = False

prev_gray = None
motion_score = 0

print("Challenge Sequence:", challenge_sequence)

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)
frame_timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_timestamp += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect_for_video(mp_image, frame_timestamp)

    # ---------------- ANTI-SPOOF MOTION CHECK ----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        motion_score = np.mean(diff)
    prev_gray = gray

    if motion_score < MOTION_THRESHOLD:
        cv2.putText(frame, "No Motion Detected (Spoof?)", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ---------------- PROCESS FACE ----------------
    if result.face_landmarks and not verified and not failed:
        landmarks = result.face_landmarks[0]
        blendshapes = result.face_blendshapes[0]
        scores = {b.category_name: b.score for b in blendshapes}

        # Draw Face Mesh
        for lm in landmarks:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        current_challenge = challenge_sequence[current_step]
        time_elapsed = time.time() - step_start_time
        time_left = int(STEP_TIME_LIMIT - time_elapsed)

        if time_left <= 0:
            failed = True

        # ---- Challenge Logic ----
        challenge_passed = False

        if current_challenge == "blink":
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)

            ear = (left_ear + right_ear) / 2.0

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
            
            if ear < EAR_THRESHOLD and blink_state == "open":
                blink_state = "closed"

            elif ear > EAR_THRESHOLD and blink_state == "closed":
                blink_state = "open"
                blink_total += 1
                print("Blink detected:", blink_total)

            if blink_total >= 1:
                challenge_passed = True
                blink_total = 0

        elif current_challenge == "smile":
            smile_score = scores.get("mouthSmileLeft", 0) + \
                          scores.get("mouthSmileRight", 0)
            if smile_score > SMILE_THRESHOLD:
                challenge_passed = True

        elif current_challenge == "mouth":
            if scores.get("jawOpen", 0) > MOUTH_THRESHOLD:
                challenge_passed = True

        elif current_challenge == "turn_left":
            nose_x = landmarks[1].x
            if nose_x > 0.55:
                challenge_passed = True

        elif current_challenge == "turn_right":
            nose_x = landmarks[1].x
            if nose_x < 0.45:
                challenge_passed = True

        if challenge_passed:
            current_step += 1
            if current_step >= TOTAL_STEPS:
                verified = True
            else:
                step_start_time = time.time()

        # ---- Display ----
        cv2.putText(frame,
                    f"Step {current_step+1}/{TOTAL_STEPS}: {current_challenge}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2)

        cv2.putText(frame,
                    f"Time Left: {time_left}s",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)
        

    # ---------------- FINAL STATES ----------------
    if verified:
        cv2.putText(frame, "VERIFIED (LIVE HUMAN)",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3)

    if failed:
        cv2.putText(frame, "FAILED - TRY AGAIN",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    cv2.imshow("Advanced Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()