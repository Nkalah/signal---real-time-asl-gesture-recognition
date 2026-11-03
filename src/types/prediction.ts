/**
 * TypeScript types for SIGNAL ASL gesture recognition.
 */

export interface PreprocessingConfig {
  version: string;
  input_features: number;
  reference_point: string;
  normalization: string;
  epsilon: number;
}

export interface ModelMetadata {
  model_name: string;
  architecture: string;
  classes_count: number;
  supported_letters: string[];
  excluded_letters: string[];
  exclusion_reason: string;
  preprocessing: PreprocessingConfig;
  input_dimension: number;
  version: string;
  training_data_source?: string;
}

export interface SystemHealth {
  status: string;
  version: string;
  service: string;
  backend_mode?: string;
  fastapi_service?: string;
}

export interface SystemReady {
  ready: boolean;
  model_loaded: boolean;
  classes_count: number;
  device: string;
  backend?: string;
  message: string;
}

export interface TopPrediction {
  letter: string;
  confidence: number;
}

export interface PredictionResult {
  ready: boolean;
  letter: string | null;
  confidence: number;
  probabilities?: Record<string, number>;
  top_predictions?: TopPrediction[];
  model_version: string;
  inference_time_ms: number;
  model_forward_time_ms?: number;
  error?: string | null;
}

export interface PipelineTimings {
  mediaPipeMs: number;
  preprocessingMs: number;
  serializationMs: number;
  apiRoundTripMs: number;
  modelForwardMs: number;
  backendInferenceMs: number;
}

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export type HandDetectionState =
  | 'idle'
  | 'initializing'
  | 'camera_ready'
  | 'detecting'
  | 'hand_detected'
  | 'no_hand'
  | 'camera_error'
  | 'permission_denied';
