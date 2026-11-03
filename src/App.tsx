/**
 * SIGNAL - Real-Time ASL Gesture Recognition
 * Core Application Component
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera,
  Activity,
  BookOpen,
  Cpu,
  Info,
  Shield,
  Layers,
  Sparkles,
} from 'lucide-react';

import {
  SystemHealth,
  SystemReady,
  ModelMetadata,
  PredictionResult,
  LandmarkPoint,
  PipelineTimings,
} from './types/prediction';
import {
  fetchHealth,
  fetchReady,
  fetchMetadata,
  predictGesture,
  preprocessLandmarks,
} from './lib/api';
import { useHandLandmarker } from './hooks/useHandLandmarker';
import { AutoCommitGate, PredictionSmoother } from './lib/predictionSmoother';
import {
  appendLetter,
  appendRecognizedLetter,
  appendSpace,
  removeLastCharacter,
} from './lib/textComposer';

import { StatusBadge } from './components/StatusBadge';
import { CameraPanel } from './components/CameraPanel';
import { PredictionCard } from './components/PredictionCard';
import { TextComposer } from './components/TextComposer';
import { PerformancePanel } from './components/PerformancePanel';
import { AlphabetReferenceModal } from './components/AlphabetReferenceModal';
import { ArchitectureModal } from './components/ArchitectureModal';

export default function App() {
  // System & API states
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [ready, setReady] = useState<SystemReady | null>(null);
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [backendStatus, setBackendStatus] = useState<'active' | 'warning' | 'error' | 'inactive'>('inactive');

  // Prediction states
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [inferenceLatency, setInferenceLatency] = useState<number>(0);
  const [pipelineTimings, setPipelineTimings] = useState<PipelineTimings>({
    mediaPipeMs: 0,
    preprocessingMs: 0,
    serializationMs: 0,
    apiRoundTripMs: 0,
    modelForwardMs: 0,
    backendInferenceMs: 0,
  });
  const [isStable, setIsStable] = useState<boolean>(false);
  const [showSkeleton, setShowSkeleton] = useState<boolean>(true);

  // Text composition states
  const [composedText, setComposedText] = useState<string>('');
  const [autoCommit, setAutoCommit] = useState<boolean>(false);

  // Modals
  const [isAlphabetOpen, setIsAlphabetOpen] = useState<boolean>(false);
  const [isArchOpen, setIsArchOpen] = useState<boolean>(false);

  // Temporal stabilization history
  const predictionSmootherRef = useRef(new PredictionSmoother());
  const requestInFlightRef = useRef<boolean>(false);
  const requestGenerationRef = useRef<number>(0);
  const stableLetterRef = useRef<string | null>(null);
  const stableSinceRef = useRef<number | null>(null);
  const autoCommitGateRef = useRef(new AutoCommitGate());

  // Configuration
  const confidenceThreshold = 0.60;

  // Poll system health and readiness on mount
  useEffect(() => {
    let isMounted = true;

    async function checkSystem() {
      try {
        const [h, r, m] = await Promise.all([
          fetchHealth().catch(() => null),
          fetchReady().catch(() => null),
          fetchMetadata().catch(() => null),
        ]);

        if (!isMounted) return;

        if (h) {
          setHealth(h);
          setBackendStatus('active');
        } else {
          setBackendStatus('error');
        }

        if (r) setReady(r);
        if (m) setMetadata(m);
      } catch {
        if (isMounted) setBackendStatus('error');
      }
    }

    checkSystem();
    const interval = setInterval(checkSystem, 8000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Callback when MediaPipe detects 21 hand landmarks in live frame
  const handleLandmarksDetected = useCallback(
    async (landmarks: LandmarkPoint[], mediaPipeMs: number) => {
      if (requestInFlightRef.current) return;
      requestInFlightRef.current = true;
      const requestGeneration = requestGenerationRef.current;

      try {
        const preprocessingStart = performance.now();
        const preprocessed = preprocessLandmarks(landmarks);
        const preprocessingMs = performance.now() - preprocessingStart;

        const { result, timings } = await predictGesture(preprocessed);
        if (requestGeneration !== requestGenerationRef.current) return;

        const now = performance.now();
        const smoothed = predictionSmootherRef.current.update(
          result,
          now,
          confidenceThreshold,
        );
        setCurrentPrediction(smoothed.prediction);
        setIsStable(smoothed.isStable);

        const measuredTimings: PipelineTimings = {
          mediaPipeMs,
          preprocessingMs,
          serializationMs: timings.serializationMs,
          apiRoundTripMs: timings.apiRoundTripMs,
          modelForwardMs:
            result.model_forward_time_ms ?? result.inference_time_ms,
          backendInferenceMs: result.inference_time_ms,
        };
        setPipelineTimings(measuredTimings);
        setInferenceLatency(
          mediaPipeMs +
            preprocessingMs +
            timings.serializationMs +
            timings.apiRoundTripMs,
        );

        const letter = smoothed.prediction.letter;
        if (smoothed.isStable && letter && smoothed.stableSince !== null) {
          if (stableLetterRef.current !== letter) {
            stableLetterRef.current = letter;
            stableSinceRef.current = smoothed.stableSince;
          }

          if (
            stableSinceRef.current !== null &&
            autoCommitGateRef.current.shouldCommit(
              letter,
              stableSinceRef.current,
              now,
              autoCommit,
            )
          ) {
            console.info('[Composer] Auto-committing stable letter', {
              letter,
              confidence: smoothed.prediction.confidence,
            });
            setComposedText((prev) => appendLetter(prev, letter));
          }
        } else {
          stableLetterRef.current = null;
          stableSinceRef.current = null;
        }

        console.debug('[Prediction]', {
          letter: letter ?? 'Uncertain',
          confidence: smoothed.prediction.confidence,
          stable: smoothed.isStable,
        });
      } catch (err: any) {
        console.error('[Inference] Prediction request error:', err);
      } finally {
        requestInFlightRef.current = false;
      }
    },
    [autoCommit, confidenceThreshold]
  );

  // MediaPipe hook for video capture and hand landmark detection
  const {
    videoRef,
    canvasRef,
    detectionState,
    errorMessage: cameraError,
    isCameraActive,
    fps,
    startCamera,
    stopCamera,
  } = useHandLandmarker(handleLandmarksDetected);

  // Invalidate pending requests whenever tracking is not producing a hand.
  useEffect(() => {
    if (detectionState !== 'hand_detected') {
      requestGenerationRef.current++;
      setCurrentPrediction(null);
      setIsStable(false);
      predictionSmootherRef.current.reset();
      stableLetterRef.current = null;
      stableSinceRef.current = null;
      autoCommitGateRef.current.reset();
    }
  }, [detectionState]);

  useEffect(() => {
    console.info('[Composer] Text state changed', {
      text: composedText,
      characters: composedText.length,
      words: composedText.trim() ? composedText.trim().split(/\s+/).length : 0,
    });
  }, [composedText]);

  // Text composer actions
  const handleAddLetter = useCallback(() => {
    const letter = currentPrediction?.letter;
    console.info('[Composer] Add Letter clicked', {
      letter,
      confidence: currentPrediction?.confidence ?? 0,
      stable: isStable,
    });
    if (letter && currentPrediction) {
      setComposedText((prev) =>
        appendRecognizedLetter(
          prev,
          letter,
          currentPrediction.confidence,
          confidenceThreshold,
          isStable,
        ),
      );
      autoCommitGateRef.current.markCommitted(letter);
    }
  }, [currentPrediction, confidenceThreshold, isStable]);

  const handleAddSpace = useCallback(() => {
    console.info('[Composer] Space clicked');
    setComposedText(appendSpace);
  }, []);

  const handleBackspace = useCallback(() => {
    console.info('[Composer] Backspace clicked');
    setComposedText(removeLastCharacter);
  }, []);

  const handleClear = useCallback(() => {
    console.info('[Composer] Clear clicked');
    setComposedText('');
  }, []);

  const handleToggleAutoCommit = useCallback(() => {
    setAutoCommit((enabled) => {
      console.info('[Composer] Auto-Commit toggled', { enabled: !enabled });
      return !enabled;
    });
  }, []);

  // Global Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if typing in an input
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) return;

      if (e.key === 'Enter') {
        e.preventDefault();
        handleAddLetter();
      } else if (e.key === ' ') {
        e.preventDefault();
        handleAddSpace();
      } else if (e.key === 'Backspace') {
        handleBackspace();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleAddLetter, handleAddSpace, handleBackspace]);

  return (
    <div className="min-h-screen bg-[#0b0f17] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* 1. Header (Strict Top Bar Contract: Brand - Status Telemetry - Actions) */}
      <header className="sticky top-0 z-40 bg-[#0d121c]/95 backdrop-blur-md border-b border-slate-800/80 px-6 py-3.5 flex items-center justify-between gap-8">
        {/* Brand Zone */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="w-8 h-8 rounded-lg bg-cyan-400 flex items-center justify-center text-slate-950 font-bold font-mono text-base shadow-sm">
            S
          </div>
          <div className="flex flex-col">
            <span className="text-base font-bold tracking-tight text-white font-mono flex items-center gap-2">
              SIGNAL
              <span className="text-xs font-normal text-cyan-400 font-sans tracking-normal px-2 py-0.5 bg-cyan-950/80 border border-cyan-800 rounded">
                ASL Alphabet Recognition
              </span>
            </span>
          </div>
        </div>

        {/* Status Telemetry Ribbon */}
        <div className="hidden md:flex items-center gap-6 shrink-0">
          <StatusBadge
            label="Camera"
            status={isCameraActive ? 'active' : 'inactive'}
            text={isCameraActive ? `${fps} FPS` : 'Off'}
          />
          <span className="text-slate-800">·</span>
          <StatusBadge
            label="Backend"
            status={backendStatus}
            text={backendStatus === 'active' ? 'Connected' : 'Offline'}
          />
          <span className="text-slate-800">·</span>
          <StatusBadge
            label="PyTorch Model"
            status={ready?.model_loaded ? 'active' : 'warning'}
            text={ready?.model_loaded ? 'Ready (24 Classes)' : 'Initializing'}
          />
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5 shrink-0">
          <button
            onClick={() => setIsAlphabetOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg transition-colors whitespace-nowrap"
          >
            <BookOpen size={14} className="text-cyan-400" />
            <span>Alphabet Guide</span>
          </button>
          <button
            onClick={() => setIsArchOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg transition-colors whitespace-nowrap"
          >
            <Cpu size={14} className="text-indigo-400" />
            <span>Architecture</span>
          </button>
        </div>
      </header>

      {/* 2. Main Workspace Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 flex flex-col gap-6">
        {/* Banner notification for J/Z static limitation */}
        <div className="px-4 py-2.5 bg-slate-900/60 rounded-xl border border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <Info size={15} className="text-cyan-400 shrink-0" />
            <span>
              Real-time assistive communication prototype classifying 24 static American Sign Language letters using local 63-coordinate hand landmark ML inference.
            </span>
          </div>
          <div className="text-[11px] font-mono text-slate-500">
            J & Z excluded (motion trajectories) · Zero data uploaded to external servers
          </div>
        </div>

        {/* 2-Column Responsive Console Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Camera + Composed Text (7 Columns) */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            {/* Live Camera Feed */}
            <CameraPanel
              videoRef={videoRef}
              canvasRef={canvasRef}
              isCameraActive={isCameraActive}
              detectionState={detectionState}
              errorMessage={cameraError}
              fps={fps}
              inferenceMs={inferenceLatency}
              onStartCamera={startCamera}
              onStopCamera={stopCamera}
              showOverlay={showSkeleton}
              onToggleOverlay={() => setShowSkeleton(!showSkeleton)}
            />

            {/* Composed Text Card */}
            <TextComposer
              currentText={composedText}
              currentLetter={
                isStable &&
                currentPrediction?.confidence &&
                currentPrediction.confidence >= confidenceThreshold
                  ? currentPrediction.letter
                  : null
              }
              isStable={isStable}
              onAddLetter={handleAddLetter}
              onAddSpace={handleAddSpace}
              onBackspace={handleBackspace}
              onClear={handleClear}
              autoCommit={autoCommit}
              onToggleAutoCommit={handleToggleAutoCommit}
            />
          </div>

          {/* Right Column: Prediction + Specifications (5 Columns) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            {/* Current Prediction Display */}
            <PredictionCard
              prediction={currentPrediction}
              detectionState={detectionState}
              isStable={isStable}
              confidenceThreshold={confidenceThreshold}
            />

            {/* Model Status & Performance Panel */}
            <PerformancePanel
              readyInfo={ready}
              metadata={metadata}
              latencyMs={inferenceLatency}
              timings={pipelineTimings}
              onOpenArchitecture={() => setIsArchOpen(true)}
              onOpenAlphabet={() => setIsAlphabetOpen(true)}
            />

            {/* Quick Practice Letters Card */}
            <div className="p-4 bg-[#111622] rounded-xl border border-slate-800/80 text-xs">
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono uppercase font-bold text-slate-400 tracking-wider text-[11px]">
                  Quick Practice Suggestions
                </span>
                <span className="text-[11px] text-slate-500 font-mono">
                  Distinctive Poses
                </span>
              </div>
              <div className="grid grid-cols-6 gap-2">
                {['A', 'B', 'C', 'L', 'V', 'Y'].map((char) => (
                  <button
                    key={char}
                    onClick={() => setIsAlphabetOpen(true)}
                    className="p-2 bg-[#0a0d14] hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/50 rounded-lg text-center transition-colors group"
                  >
                    <span className="font-mono text-lg font-bold text-slate-200 group-hover:text-cyan-400 block">
                      {char}
                    </span>
                    <span className="text-[10px] text-slate-500">Sign</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* 3. Footer */}
      <footer className="mt-auto border-t border-slate-900 bg-[#090d14] px-6 py-4 text-xs text-slate-500 flex flex-wrap items-center justify-between gap-4 font-mono">
        <div className="flex items-center gap-2">
          <span>SIGNAL ASL v1.0.0</span>
          <span>·</span>
          <span>PyTorch MLP (63-128-64-24)</span>
          <span>·</span>
          <span>MediaPipe Tasks Vision</span>
        </div>
        <div className="flex items-center gap-4 text-[11px]">
          <span>Privacy: Local Browser Inference</span>
          <span>·</span>
          <span>24 Static Alphabet Classes</span>
        </div>
      </footer>

      {/* 4. Modals */}
      <AlphabetReferenceModal
        isOpen={isAlphabetOpen}
        onClose={() => setIsAlphabetOpen(false)}
      />

      <ArchitectureModal
        isOpen={isArchOpen}
        onClose={() => setIsArchOpen(false)}
      />
    </div>
  );
}
