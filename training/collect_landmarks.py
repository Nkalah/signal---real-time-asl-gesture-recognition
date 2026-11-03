"""Interactive Landmark Collection utility for recording ASL training data."""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.config import SUPPORTED_LETTERS, PREPROCESSING_VERSION
from training.prepare_data import normalize_landmarks


def collect_webcam_samples(letter: str, participant_id: str, num_samples: int = 50, output_dir: Path = Path("data/raw")):
    """
    Collects hand landmarks using MediaPipe and OpenCV from a live webcam feed.
    """
    if letter not in SUPPORTED_LETTERS:
        print(f"Error: Letter '{letter}' is not one of the 24 supported static classes: {SUPPORTED_LETTERS}")
        print("Note: 'J' and 'Z' require dynamic movement and cannot be captured as static poses.")
        return

    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("Error: cv2 or mediapipe not installed in current environment.")
        print("Please run: pip install mediapipe opencv-python-headless")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / f"landmarks_{letter}_{participant_id}_{int(time.time())}.npy"

    print(f"\nReady to collect {num_samples} samples for letter '{letter}' (Participant: {participant_id})")
    print("Press 'c' to capture a sample, or 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    collected_samples = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:
        while len(collected_samples) < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Display instructions
            cv2.putText(
                frame,
                f"Signing: {letter} | Collected: {len(collected_samples)}/{num_samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.append([lm.x, lm.y, lm.z])
                coords_np = np.array(coords, dtype=np.float32)

                # Capture on 'c' key or auto
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    normalized = normalize_landmarks(coords_np)
                    collected_samples.append(normalized)
                    print(f"Captured {len(collected_samples)}/{num_samples}")
                elif key == ord('q'):
                    break
            else:
                cv2.putText(frame, "No Hand Detected", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            cv2.imshow("SIGNAL ASL Data Collector", frame)

    cap.release()
    cv2.destroyAllWindows()

    if collected_samples:
        arr = np.array(collected_samples, dtype=np.float32)
        np.save(save_file, arr)
        print(f"\nSuccessfully saved {len(collected_samples)} samples to {save_file}")
    else:
        print("\nNo samples were captured.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect labeled ASL hand landmarks from webcam.")
    parser.add_argument("--letter", type=str, default="A", help="Target ASL letter (A-Y, excluding J and Z)")
    parser.add_argument("--participant", type=str, default="user_1", help="Participant / Session identifier")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to collect")
    args = parser.parse_args()

    collect_webcam_samples(letter=args.letter.upper(), participant_id=args.participant, num_samples=args.samples)
