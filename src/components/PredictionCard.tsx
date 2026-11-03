import React from 'react';
import { PredictionResult, HandDetectionState } from '../types/prediction';

interface PredictionCardProps {
  prediction: PredictionResult | null;
  detectionState: HandDetectionState;
  isStable: boolean;
  confidenceThreshold: number;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  prediction,
  detectionState,
  isStable,
  confidenceThreshold,
}) => {
  const hasHand = detectionState === 'hand_detected';
  const confidence = prediction?.confidence ?? 0;
  const letter = prediction?.letter ?? null;
  const isHighConfidence = confidence >= confidenceThreshold;

  // Percentage value
  const confidencePct = Math.round(confidence * 100);

  return (
    <div className="flex flex-col bg-[#111622] rounded-xl border border-slate-800/80 p-5 shadow-xl">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-4">
        <div>
          <h2 className="text-xs uppercase font-mono font-semibold tracking-wider text-slate-400">
            Current Prediction
          </h2>
          <p className="text-[11px] text-slate-500">
            Local landmark model inference
          </p>
        </div>

        {/* Stability indicator */}
        <div className="flex items-center gap-1.5 text-xs font-mono">
          {hasHand && letter && isHighConfidence && isStable ? (
            <span className="inline-flex items-center gap-1 font-semibold text-emerald-400">
              <span className="text-[10px]">●</span>
              <span>STABLE</span>
            </span>
          ) : (
            <span className="text-slate-500">
              {hasHand ? 'UNCERTAIN' : 'WAITING'}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 items-center">
        {/* Large Current Letter Display */}
        <div className="flex flex-col items-center justify-center p-4 bg-[#0a0d14] rounded-lg border border-slate-800/90 min-h-[130px]">
          {hasHand && letter && isHighConfidence && isStable ? (
            <div className="flex flex-col items-center">
              <span className="text-6xl font-bold font-mono tracking-tight text-white drop-shadow-md">
                {letter}
              </span>
              <span className="text-[11px] text-slate-400 font-mono mt-1">
                ASL Letter {letter}
              </span>
            </div>
          ) : hasHand ? (
            <div className="flex flex-col items-center text-center">
              <span className="text-3xl font-mono text-slate-500 font-bold">
                {letter && isHighConfidence ? `${letter}…` : '?'}
              </span>
              <span className="text-[11px] text-amber-400 font-mono mt-1">
                {letter && isHighConfidence ? 'Stabilizing' : 'Uncertain'}
              </span>
            </div>
          ) : (
            <div className="flex flex-col items-center text-center text-slate-600">
              <span className="text-5xl font-mono font-light">--</span>
              <span className="text-[11px] mt-1 font-mono">No Hand Present</span>
            </div>
          )}
        </div>

        {/* Confidence & Statistics */}
        <div className="flex flex-col justify-between h-full py-1">
          <div>
            <div className="flex items-baseline justify-between mb-1.5">
              <span className="text-xs text-slate-400 font-medium">Confidence</span>
              <span className="text-2xl font-mono font-bold text-slate-100 tabular-nums">
                {hasHand ? `${confidencePct}%` : '0%'}
              </span>
            </div>

            {/* Custom Meter */}
            <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-150 ${
                  letter && confidence >= confidenceThreshold && isStable
                    ? 'bg-gradient-to-r from-cyan-500 to-emerald-400'
                    : 'bg-amber-500'
                }`}
                style={{ width: `${hasHand ? confidencePct : 0}%` }}
              />
            </div>

            <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
              <span>Threshold: {Math.round(confidenceThreshold * 100)}%</span>
              <span>
                {!hasHand
                  ? 'No hand'
                  : !isHighConfidence
                    ? 'Uncertain'
                    : isStable
                      ? 'Stable'
                      : 'Stabilizing'}
              </span>
            </div>
          </div>

          {/* Top-3 Alternatives */}
          <div className="mt-3 pt-3 border-t border-slate-800/60">
            <span className="text-[10px] uppercase font-mono text-slate-500 block mb-1">
              Top Ranked Classes
            </span>
            <div className="flex items-center gap-2">
              {prediction?.top_predictions && prediction.top_predictions.length > 0 ? (
                prediction.top_predictions.map((item, idx) => (
                  <div
                    key={idx}
                    className="flex-1 px-2 py-1 bg-slate-900/90 rounded border border-slate-800 text-center font-mono"
                  >
                    <span className="text-xs font-bold text-slate-200">
                      {item.letter}
                    </span>
                    <span className="text-[10px] text-slate-500 block tabular-nums">
                      {Math.round(item.confidence * 100)}%
                    </span>
                  </div>
                ))
              ) : (
                <span className="text-xs text-slate-600 font-mono">None</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
