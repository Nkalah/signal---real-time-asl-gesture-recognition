/**
 * Full-stack Express server with Vite integration and ASL API endpoints.
 */

import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = parseInt(process.env.PORT || '3000', 10);
const FASTAPI_PORT = parseInt(process.env.FASTAPI_PORT || '8008', 10);
const FASTAPI_URL = `http://127.0.0.1:${FASTAPI_PORT}`;

app.use(express.json({ limit: '1mb' }));

// Supported classes: 24 static letters
const SUPPORTED_LETTERS = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
  'T', 'U', 'V', 'W', 'X', 'Y'
];
const EXCLUDED_LETTERS = ['J', 'Z'];

// Load model weights and config from disk
let modelWeights: { [key: string]: number[][] | number[] } | null = null;
let modelConfig: any = null;

try {
  const configPath = path.resolve(__dirname, 'backend/models/config.json');
  const weightsPath = path.resolve(__dirname, 'backend/models/weights.json');
  if (fs.existsSync(configPath)) {
    modelConfig = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  }
  if (fs.existsSync(weightsPath)) {
    modelWeights = JSON.parse(fs.readFileSync(weightsPath, 'utf-8'));
    console.log('[Server] Loaded trained ASL MLP weights from backend/models/weights.json');
  }
} catch (err) {
  console.warn('[Server] Note: Trained weights file not yet loaded:', err);
}

// Forward pass implementation of ASLLandmarkMLP using trained PyTorch weights
function runLocalInference(landmarks: number[]): {
  letter: string;
  confidence: number;
  top_predictions: { letter: string; confidence: number }[];
  probabilities: { [key: string]: number };
  model_forward_time_ms: number;
} {
  if (!modelWeights) {
    throw new Error('Model weights not loaded on server.');
  }

  const w1 = modelWeights['fc1.weight'] as number[][]; // [128, 63]
  const b1 = modelWeights['fc1.bias'] as number[];     // [128]
  const w2 = modelWeights['fc2.weight'] as number[][]; // [64, 128]
  const b2 = modelWeights['fc2.bias'] as number[];     // [64]
  const w3 = modelWeights['fc3.weight'] as number[][]; // [24, 64]
  const b3 = modelWeights['fc3.bias'] as number[];     // [24]

  const modelForwardStart = performance.now();

  // Layer 1: Linear(63 -> 128) + ReLU
  const h1 = new Float32Array(128);
  for (let i = 0; i < 128; i++) {
    let sum = b1[i];
    const row = w1[i];
    for (let j = 0; j < 63; j++) {
      sum += row[j] * landmarks[j];
    }
    h1[i] = Math.max(0, sum); // ReLU
  }

  // Layer 2: Linear(128 -> 64) + ReLU
  const h2 = new Float32Array(64);
  for (let i = 0; i < 64; i++) {
    let sum = b2[i];
    const row = w2[i];
    for (let j = 0; j < 128; j++) {
      sum += row[j] * h1[j];
    }
    h2[i] = Math.max(0, sum); // ReLU
  }

  // Layer 3: Linear(64 -> 24)
  const logits = new Float32Array(24);
  let maxLogit = -Infinity;
  for (let i = 0; i < 24; i++) {
    let sum = b3[i];
    const row = w3[i];
    for (let j = 0; j < 64; j++) {
      sum += row[j] * h2[j];
    }
    logits[i] = sum;
    if (sum > maxLogit) maxLogit = sum;
  }
  const modelForwardMs = performance.now() - modelForwardStart;

  // Softmax
  let expSum = 0;
  const exps = new Float32Array(24);
  for (let i = 0; i < 24; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    exps[i] = e;
    expSum += e;
  }

  const probs = new Float32Array(24);
  let bestIdx = 0;
  let bestProb = 0;
  const probsObj: { [key: string]: number } = {};

  for (let i = 0; i < 24; i++) {
    probs[i] = exps[i] / (expSum || 1);
    const letter = SUPPORTED_LETTERS[i];
    probsObj[letter] = probs[i];
    if (probs[i] > bestProb) {
      bestProb = probs[i];
      bestIdx = i;
    }
  }

  // Top predictions sorted
  const ranked = SUPPORTED_LETTERS.map((letter, idx) => ({
    letter,
    confidence: probs[idx],
  })).sort((a, b) => b.confidence - a.confidence);

  return {
    letter: SUPPORTED_LETTERS[bestIdx],
    confidence: bestProb,
    top_predictions: ranked.slice(0, 3),
    probabilities: probsObj,
    model_forward_time_ms: modelForwardMs,
  };
}

// 1. Diagnostics: GET /api/health
app.get('/api/health', async (_req, res) => {
  // Check if FastAPI is responsive
  let fastApiStatus = 'offline';
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 800);
    const resp = await fetch(`${FASTAPI_URL}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    if (resp.ok) fastApiStatus = 'healthy';
  } catch {}

  res.json({
    status: 'healthy',
    version: '1.0.0',
    service: 'SIGNAL ASL Gesture Recognition Service',
    backend_mode: fastApiStatus === 'healthy' ? 'fastapi_proxy' : 'direct_inference',
    fastapi_service: fastApiStatus,
  });
});

// 2. Readiness: GET /api/ready
app.get('/api/ready', async (_req, res) => {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 800);
    const resp = await fetch(`${FASTAPI_URL}/ready`, { signal: controller.signal });
    clearTimeout(timeout);
    if (resp.ok) {
      const data = await resp.json();
      return res.json(data);
    }
  } catch {}

  const modelLoaded = modelWeights !== null;
  res.json({
    ready: modelLoaded,
    model_loaded: modelLoaded,
    classes_count: SUPPORTED_LETTERS.length,
    device: 'cpu',
    backend: 'SIGNAL Core Inference Engine',
    message: modelLoaded
      ? 'Model is loaded and ready for real-time inference.'
      : 'Model weights not found. Please train model first.',
  });
});

// 3. Metadata: GET /api/metadata
app.get('/api/metadata', async (_req, res) => {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 800);
    const resp = await fetch(`${FASTAPI_URL}/metadata`, { signal: controller.signal });
    clearTimeout(timeout);
    if (resp.ok) {
      const data = await resp.json();
      return res.json(data);
    }
  } catch {}

  res.json({
    model_name: 'SIGNAL ASL Landmark MLP',
    architecture: 'MLP-63-128-64-24',
    classes_count: 24,
    supported_letters: SUPPORTED_LETTERS,
    excluded_letters: EXCLUDED_LETTERS,
    exclusion_reason:
      'Letters J and Z require motion trajectories and are not reliably recognizable from static 3D landmark poses.',
    preprocessing: {
      version: 'v1.0.0',
      input_features: 63,
      reference_point: 'wrist (landmark 0)',
      normalization: "p_i' = (p_i - p_0) / (max_dist + epsilon)",
      epsilon: 1e-6,
    },
    input_dimension: 63,
    version: '1.0.0',
    training_data_source: modelConfig?.training_metadata?.dataset_source || 'unknown',
  });
});

// 4. Inference: POST /api/predict
app.post('/api/predict', async (req, res) => {
  const { landmarks } = req.body || {};

  // Validation
  if (!landmarks || !Array.isArray(landmarks)) {
    return res.status(422).json({
      detail: 'Request body must contain "landmarks" array of numbers.',
    });
  }

  if (landmarks.length !== 63) {
    return res.status(422).json({
      detail: `Expected exactly 63 normalized coordinates (21 landmarks x 3), received ${landmarks.length}`,
    });
  }

  for (let i = 0; i < landmarks.length; i++) {
    const v = landmarks[i];
    if (typeof v !== 'number' || isNaN(v) || !isFinite(v)) {
      return res.status(422).json({
        detail: `Landmark at index ${i} contains non-finite value: ${v}`,
      });
    }
    if (Math.abs(v) > 10.0) {
      return res.status(422).json({
        detail: `Normalized landmark at index ${i} exceeds expected bounds: ${v}`,
      });
    }
  }

  const startTime = performance.now();

  // Try proxying to FastAPI if active
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1200);
    const resp = await fetch(`${FASTAPI_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks }),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (resp.ok) {
      const data = await resp.json();
      return res.json(data);
    }
  } catch {}

  // Fallback / direct inference using exact PyTorch-trained weights
  try {
    if (!modelWeights) {
      return res.status(503).json({
        detail: 'Model artifact not loaded. Please ensure training workflow was completed.',
      });
    }

    const result = runLocalInference(landmarks);
    const duration = performance.now() - startTime;

    return res.json({
      ready: true,
      letter: result.letter,
      confidence: result.confidence,
      probabilities: result.probabilities,
      top_predictions: result.top_predictions,
      model_version: '1.0.0',
      inference_time_ms: parseFloat(duration.toFixed(2)),
      model_forward_time_ms: parseFloat(result.model_forward_time_ms.toFixed(2)),
      error: null,
    });
  } catch (err: any) {
    return res.status(422).json({
      detail: `Inference error: ${err?.message || err}`,
    });
  }
});

// Configure Vite middlewares in dev mode or static files in production
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Server] SIGNAL ASL Full-Stack Server running at http://0.0.0.0:${PORT}`);
  });
}

startServer();
