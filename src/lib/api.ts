/**
 * API client and preprocessing utilities for SIGNAL ASL recognition.
 */

import {
  SystemHealth,
  SystemReady,
  ModelMetadata,
  PredictionResult,
  LandmarkPoint,
} from '../types/prediction';

const API_BASE = '/api';

/**
 * Standard Landmark Preprocessing Specification v1.0.0
 *
 * 1. Takes 21 landmark points [{x, y, z}, ...]
 * 2. Checks all points are valid and finite
 * 3. Shifts relative to wrist (point 0): p_i' = p_i - p_0
 * 4. Normalizes by hand scale s = max(||p_i'||): p_i'' = p_i' / (s + 1e-6)
 * 5. Returns flattened 63-element float array
 */
export function preprocessLandmarks(landmarks: LandmarkPoint[]): number[] {
  if (!landmarks || landmarks.length !== 21) {
    throw new Error(`Expected exactly 21 landmarks, received ${landmarks?.length || 0}`);
  }

  const wrist = landmarks[0];
  if (![wrist.x, wrist.y, wrist.z].every(Number.isFinite)) {
    throw new Error('Invalid wrist landmark');
  }

  let maxDist = 0;
  const shifted: { x: number; y: number; z: number }[] = [];

  for (let i = 0; i < 21; i++) {
    const lm = landmarks[i];
    if (![lm.x, lm.y, lm.z].every(Number.isFinite)) {
      throw new Error(`Invalid landmark at index ${i}`);
    }

    const dx = lm.x - wrist.x;
    const dy = lm.y - wrist.y;
    const dz = lm.z - wrist.z;
    shifted.push({ x: dx, y: dy, z: dz });

    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > maxDist) maxDist = dist;
  }

  const epsilon = 1e-6;
  if (maxDist < epsilon) {
    throw new Error(`Degenerate hand scale detected: ${maxDist}`);
  }
  const scale = maxDist + epsilon;

  const vector: number[] = [];
  for (let i = 0; i < 21; i++) {
    vector.push(
      shifted[i].x / scale,
      shifted[i].y / scale,
      shifted[i].z / scale
    );
  }

  return vector;
}

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export async function fetchReady(): Promise<SystemReady> {
  const res = await fetch(`${API_BASE}/ready`);
  if (!res.ok) throw new Error(`Readiness check failed (${res.status})`);
  return res.json();
}

export async function fetchMetadata(): Promise<ModelMetadata> {
  const res = await fetch(`${API_BASE}/metadata`);
  if (!res.ok) throw new Error(`Metadata check failed (${res.status})`);
  return res.json();
}

export interface PredictionRequestTimings {
  serializationMs: number;
  apiRoundTripMs: number;
}

export async function predictGesture(
  normalizedLandmarks: number[]
): Promise<{ result: PredictionResult; timings: PredictionRequestTimings }> {
  const serializationStart = performance.now();
  const body = JSON.stringify({ landmarks: normalizedLandmarks });
  const serializationMs = performance.now() - serializationStart;
  const apiStart = performance.now();
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 5000);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      signal: controller.signal,
    });

    if (!res.ok) {
      const errorBody = await res.text();
      throw new Error(errorBody || `Prediction failed with status ${res.status}`);
    }

    const result: PredictionResult = await res.json();
    return {
      result,
      timings: {
        serializationMs,
        apiRoundTripMs: performance.now() - apiStart,
      },
    };
  } finally {
    window.clearTimeout(timeoutId);
  }
}
