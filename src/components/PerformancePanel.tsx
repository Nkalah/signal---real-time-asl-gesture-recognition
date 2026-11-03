import React from 'react';
import { Cpu, Activity, AlertTriangle, ShieldCheck, Database } from 'lucide-react';
import { ModelMetadata, PipelineTimings, SystemReady } from '../types/prediction';

interface PerformancePanelProps {
  readyInfo: SystemReady | null;
  metadata: ModelMetadata | null;
  latencyMs: number;
  timings: PipelineTimings;
  onOpenArchitecture: () => void;
  onOpenAlphabet: () => void;
}

export const PerformancePanel: React.FC<PerformancePanelProps> = ({
  readyInfo,
  metadata,
  latencyMs,
  timings,
  onOpenArchitecture,
  onOpenAlphabet,
}) => {
  return (
    <div className="flex flex-col bg-[#111622] rounded-xl border border-slate-800/80 p-5 shadow-xl">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-4">
        <div>
          <h2 className="text-xs uppercase font-mono font-semibold tracking-wider text-slate-400">
            Model Specifications & Diagnostics
          </h2>
          <p className="text-[11px] text-slate-500">
            Local landmark model inference
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenArchitecture}
            className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 underline"
          >
            Architecture
          </button>
          <span className="text-slate-700">·</span>
          <button
            onClick={onOpenAlphabet}
            className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 underline"
          >
            Alphabet Guide
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
        {/* Model State */}
        <div className="p-3 bg-[#0a0d14] rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 mb-1">
            <ShieldCheck size={14} className="text-emerald-400" />
            <span className="text-[11px] uppercase tracking-wider font-sans">Model Status</span>
          </div>
          <span className="text-slate-200 font-bold block">
            {readyInfo?.model_loaded ? 'Model Loaded & Active' : 'Loading Model...'}
          </span>
          <span className="text-[10px] text-slate-500">
            Device: {readyInfo?.device || 'CPU'}
          </span>
        </div>

        {/* Classes */}
        <div className="p-3 bg-[#0a0d14] rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 mb-1">
            <Database size={14} className="text-cyan-400" />
            <span className="text-[11px] uppercase tracking-wider font-sans">Supported Classes</span>
          </div>
          <span className="text-slate-200 font-bold block">
            24 Letters
          </span>
          <span className="text-[10px] text-slate-500">
            A–Z (excluding J, Z)
          </span>
        </div>

        {/* Feature input */}
        <div className="p-3 bg-[#0a0d14] rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 mb-1">
            <Cpu size={14} className="text-indigo-400" />
            <span className="text-[11px] uppercase tracking-wider font-sans">Input Features</span>
          </div>
          <span className="text-slate-200 font-bold block tabular-nums">
            63 Dimensions
          </span>
          <span className="text-[10px] text-slate-500">
            21 landmarks × 3 (x,y,z)
          </span>
        </div>

        {/* Latency */}
        <div className="p-3 bg-[#0a0d14] rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-400 mb-1">
            <Activity size={14} className="text-emerald-400" />
            <span className="text-[11px] uppercase tracking-wider font-sans">Pipeline Latency</span>
          </div>
          <span className="text-slate-200 font-bold block tabular-nums">
            {latencyMs > 0 ? `~${latencyMs.toFixed(1)} ms` : '--'}
          </span>
          <span className="text-[10px] text-slate-500">
            MediaPipe + preprocessing + API round-trip
          </span>
          {latencyMs > 0 && (
            <span className="text-[9px] text-slate-500 block mt-1 leading-relaxed">
              MP {timings.mediaPipeMs.toFixed(1)} · Prep {timings.preprocessingMs.toFixed(1)} ·
              JSON {timings.serializationMs.toFixed(1)} · API {timings.apiRoundTripMs.toFixed(1)} ·
              model {timings.modelForwardMs.toFixed(1)} · backend {timings.backendInferenceMs.toFixed(1)} ms
            </span>
          )}
        </div>
      </div>

      {metadata?.training_data_source === 'synthetic' && (
        <div className="mt-4 rounded-lg border border-amber-900/70 bg-amber-950/30 px-3 py-2 text-[11px] text-amber-200">
          This checkpoint was trained on synthetic poses only. Confidence is not
          real-world accuracy; collect labeled webcam samples and retrain.
        </div>
      )}

      {/* Critical Scope Limitation Notice */}
      <div className="mt-4 p-3 bg-slate-900/60 rounded-lg border border-slate-800 flex items-start gap-2 text-xs">
        <AlertTriangle size={15} className="text-amber-400 shrink-0 mt-0.5" />
        <p className="text-slate-400 leading-relaxed text-[11px]">
          <strong className="text-slate-200">Scope Limitation:</strong> Letters <strong>J</strong> and <strong>Z</strong> require dynamic tracing motions and are deliberately excluded from this static pose classifier. Full ASL also incorporates grammar and facial expressions.
        </p>
      </div>
    </div>
  );
};
