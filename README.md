# SIGNAL — Real-Time ASL Gesture Recognition

> **AI-Powered Assistive Communication Prototype**  
> Converts American Sign Language (ASL) static alphabet gestures into readable text in real-time using in-browser MediaPipe hand landmark tracking and a PyTorch MLP classification model.

---

## 1. Product Overview

SIGNAL is a local, privacy-first computer-vision web application that enables users to sign the ASL alphabet in front of an ordinary webcam and convert gestures into composed sentences.

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Lucide React, MediaPipe Tasks Vision.
- **Backend**: Python 3.10+, FastAPI, PyTorch (CPU-optimized), Uvicorn, Pydantic, Scikit-learn.
- **Model Architecture**: PyTorch Multilayer Perceptron (MLP) mapping 63 normalized coordinates to 24 static letter classes.
- **Zero API Dependency**: Completely local inference with zero frame upload to third-party cloud services.

### Scope Limitation: The 24-Class Alphabet
In American Sign Language, letters **J** and **Z** involve dynamic movement trajectories (tracing "J" with the pinky, and drawing "Z" with the index finger). Static pose classifiers cannot reliably differentiate them from resting finger positions. SIGNAL explicitly documents and supports **24 static alphabet classes**:
```
A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
```

---

## 2. Architecture & Pipeline

```
Browser Webcam Feed
    ↓
MediaPipe Tasks Vision (hand_landmarker.task)
    ↓
21 Hand Landmarks (x, y, z)
    ↓
Standard Preprocessing (v1.0.0):
  1. Wrist-relative translation: p_i' = p_i - p_0
  2. Euclidean hand-scale normalization: s = max(||p_i - p_0||)
  3. Normalized coordinate vector: p_i'' = (p_i - p_0) / (s + 1e-6)
    ↓
63 Normalized Float32 Features [x0, y0, z0, ..., x20, y20, z20]
    ↓
PyTorch MLP Classifier (Input: 63 → Linear(128) → ReLU → Dropout(0.2) → Linear(64) → ReLU → Linear(24) → Softmax)
    ↓
Validated Prediction Response (Letter, Softmax Confidence, Top 3 Candidates, Latency)
    ↓
Temporal Smoothing & Text Composition (Add Letter, Space, Backspace, Clear, Read Aloud)
```

---

## 3. Project Structure

```
asl-gesture-recognition/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI server with CORS and diagnostics
│   │   ├── schemas.py       # Pydantic request/response schemas
│   │   ├── model.py         # PyTorch ASLLandmarkMLP architecture
│   │   ├── predictor.py     # Singleton inference manager with timing
│   │   └── config.py        # Configuration, class labels, and paths
│   ├── models/
│   │   ├── asl_mlp_model.pth    # Trained PyTorch weights state_dict
│   │   ├── config.json          # Architecture and training metadata
│   │   ├── metrics_summary.json # Per-class precision, recall, and F1
│   │   └── weights.json         # Universal runtime weights export
│   ├── tests/
│   │   ├── test_health.py       # Health and metadata endpoint tests
│   │   ├── test_validation.py   # Coordinate range and length tests
│   │   └── test_prediction.py   # Real inference and accuracy tests
│   ├── requirements.txt
│   └── .env.example
├── training/
│   ├── prepare_data.py      # Preprocessing and anatomical generator
│   ├── train.py             # Reproducible PyTorch training loop
│   ├── evaluate.py          # Evaluation script with per-class report
│   └── collect_landmarks.py # Interactive webcam data collection CLI
├── data/
│   ├── raw/                 # Raw recorded landmark arrays
│   └── processed/           # Preprocessed CSV/NPY datasets
├── public/
│   ├── models/              # Local MediaPipe hand_landmarker.task
│   └── wasm/                # MediaPipe vision WASM runtime
├── src/
│   ├── components/
│   │   ├── CameraPanel.tsx
│   │   ├── PredictionCard.tsx
│   │   ├── TextComposer.tsx
│   │   ├── PerformancePanel.tsx
│   │   ├── StatusBadge.tsx
│   │   ├── AlphabetReferenceModal.tsx
│   │   └── ArchitectureModal.tsx
│   ├── hooks/
│   │   └── useHandLandmarker.ts
│   ├── lib/
│   │   └── api.ts
│   ├── types/
│   │   └── prediction.ts
│   ├── App.tsx
│   └── index.css
├── server.ts                # Express full-stack server with Vite middleware
├── package.json
└── README.md
```

---

## 4. Windows PowerShell Quick Start Guide

### Step 1: Clone and Set Up Virtual Environment

Open **PowerShell** in the project directory:

```powershell
# 1. Create Python virtual environment
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install Python backend dependencies
pip install -r backend\requirements.txt

# 4. Install Node.js frontend dependencies
npm install
```

### Step 2: Model Training & Evaluation (Reproducible)

A pre-trained model artifact is already provided in `backend/models/asl_mlp_model.pth`. Its reported 99.44% score is only on generated canonical poses, not real signers. Collect real examples before interpreting a retrained score as live accuracy.

```powershell
# Train the model
python training\train.py

# Run evaluation on held-out test data
python training\evaluate.py
```

### Step 3: Run Backend Tests

```powershell
# Execute pytest suite
$env:PYTHONPATH="."
pytest backend\tests
```

### Step 4: Start Development Servers

**Option A: Unified Full-Stack Server (Recommended)**
```powershell
npm run dev
# Serves application and API endpoints at http://localhost:3000
```

**Option B: Independent Terminals**

Terminal 1 (Python FastAPI Backend):
```powershell
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH="."
uvicorn backend.app.main:app --host 127.0.0.1 --port 8008 --reload
```

Terminal 2 (Vite Frontend):
```powershell
npm run dev
```

---

## 5. Collecting Custom Labeled Landmark Data (SIGNAL V2)

Open the browser-based collector at `http://localhost:3000/collect` while the app is running. It uses the same `@mediapipe/tasks-vision` HandLandmarker instance, model asset, and camera feed as the production app. It records raw image-coordinate landmarks (21 × x/y/z), handedness, participant ID, session ID, recording condition, and timestamp. Video frames are not saved or uploaded.

1. Start the camera and select a letter.
2. Start capture while holding the correct ASL pose. The collector samples at approximately 3 frames per second.
3. During a capture burst, make small natural changes in hand angle and distance without changing the letter.
4. Pause, lower your hand, reset the pose, and repeat in a separate take.
5. Repeat in separate sessions with different lighting, distances, angles, and backgrounds.
6. Export the CSV and place it in `data/raw/`.

The collector exports `signal_landmarks_<participant>_<session>.csv`. The training loader reads these raw coordinates and applies `normalize_landmarks()` itself, avoiding double-normalization. Legacy `.npy` collector files remain supported. Do not edit the coordinate columns manually.

**Single-participant limitation:** use the same participant ID for yourself and a new session ID for each separate recording session. With three or more sessions, the loader keeps sessions separate between train/validation/test to reduce adjacent-frame leakage. This measures performance on later sessions of the same signer; it is **not** signer-independent generalization. A real unseen-signer evaluation requires additional people.

Start with 50–100 varied samples per class to verify the full pipeline. Then build toward 300–500 per class across several sessions. Do not collect hundreds of adjacent frames from one uninterrupted hold and treat them as independent examples.

```powershell
# Run the app
npm run dev

# Open the collector in your browser
# http://localhost:3000/collect

# After exporting CSVs into data/raw/, train and evaluate
python training\train.py
python training\evaluate.py
```

The frontend test suite covers stable-prediction gating, auto-commit timing/rearming, and text operations:

```powershell
npm test
```

---

## 6. Preprocessing Specification (v1.0.0)

1. **Landmark Ordering**: Standard MediaPipe 21 landmarks (0: Wrist, 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky).
2. **Missing Coordinates**: All 21 landmarks must contain valid finite numbers. Any NaN or Inf fails validation.
3. **Wrist Translation**: $p_i' = p_i - p_0$ where $p_0$ is the wrist landmark.
4. **Scale Normalization**: $s = \max_{i} (\|p_i - p_0\|_2)$
5. **Epsilon Safeguard**: $\epsilon = 1 \times 10^{-6}$. If $s < \epsilon$, error is raised.
6. **Feature Output**: $p_i'' = \frac{p_i - p_0}{s + \epsilon}$, flattened into 63 floats.

---

## 7. Model Performance

The checked-in checkpoint's **99.44%** score was measured on a random split of augmented versions of generated canonical hand poses. It is not validation on real signers and substantially overstates live accuracy. No real collected dataset is present in the repository, so real-world per-class accuracy is currently unknown. The checked-in synthetic test report shows its few errors concentrated around M/N and R/S/T/U; that should not be read as a real-sign confusability report.

The app reports MediaPipe time, preprocessing, JSON serialization, API round-trip, and backend-reported inference time separately. API round-trip includes server and transport overhead; it is not a pure network measurement. Webcam FPS depends on the camera and local MediaPipe runtime.

| Metric | Checked-in artifact |
|---|---|
| Synthetic generated-pose test accuracy | 99.44% (not representative of real users) |
| Supported classes | 24 (A–Z excluding J and Z) |
| Input features | 63 normalized coordinates |
| Real-signer validation accuracy | Not measured |

---

## 8. Troubleshooting

- **Camera Permission Blocked**: Click the camera/lock icon in your browser's address bar and set Camera to "Allow", then refresh.
- **Model Weights Missing**: Run `python training\train.py` to re-generate `backend/models/asl_mlp_model.pth`.
- **Low Confidence Predictions**: Ensure good front lighting, position your hand 1.5 to 2.5 feet from the camera lens, and refer to the in-app **Alphabet Guide** for exact finger placement.
