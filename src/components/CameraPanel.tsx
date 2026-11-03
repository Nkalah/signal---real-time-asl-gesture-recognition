import React from 'react';
import { Camera, CameraOff, AlertCircle, RefreshCw, Eye, EyeOff } from 'lucide-react';
import { HandDetectionState } from '../types/prediction';

interface CameraPanelProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  isCameraActive: boolean;
  detectionState: HandDetectionState;
  errorMessage: string | null;
  fps: number;
  inferenceMs: number;
  onStartCamera: () => void;
  onStopCamera: () => void;
  showOverlay: boolean;
  onToggleOverlay: () => void;
}

export const CameraPanel: React.FC<CameraPanelProps> = ({
  videoRef,
  canvasRef,
  isCameraActive,
  detectionState,
  errorMessage,
  fps,
  inferenceMs,
  onStartCamera,
  onStopCamera,
  showOverlay,
  onToggleOverlay,
}) => {
  return (
    <div className="relative flex flex-col bg-[#111622] rounded-xl border border-slate-800/80 overflow-hidden shadow-2xl">
      {/* Top Console Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-[#0d121c] border-b border-slate-800/80 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-slate-200 tracking-wide uppercase font-mono text-[11px]">
            Input Video Feed
          </span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400 font-mono">
            {isCameraActive ? `${fps} FPS` : 'OFFLINE'}
          </span>
          {isCameraActive && inferenceMs > 0 && (
            <>
              <span className="text-slate-600">·</span>
              <span className="text-cyan-400 font-mono">
                {inferenceMs.toFixed(1)}ms total
              </span>
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          {isCameraActive && (
            <button
              onClick={onToggleOverlay}
              title={showOverlay ? 'Hide skeleton overlay' : 'Show skeleton overlay'}
              className="flex items-center gap-1 px-2 py-1 text-slate-400 hover:text-slate-200 bg-slate-800/60 hover:bg-slate-800 rounded text-[11px] transition-colors"
            >
              {showOverlay ? <Eye size={13} /> : <EyeOff size={13} />}
              <span>{showOverlay ? 'Overlay' : 'Clean'}</span>
            </button>
          )}

          {isCameraActive ? (
            <button
              onClick={onStopCamera}
              className="flex items-center gap-1 px-2.5 py-1 text-rose-300 hover:text-rose-100 bg-rose-950/40 hover:bg-rose-900/60 border border-rose-800/60 rounded text-[11px] font-medium transition-colors"
            >
              <CameraOff size={13} />
              <span>Stop Camera</span>
            </button>
          ) : (
            <button
              onClick={onStartCamera}
              className="flex items-center gap-1.5 px-3 py-1 text-slate-900 bg-cyan-400 hover:bg-cyan-300 font-semibold rounded text-[11px] transition-colors shadow-sm"
            >
              <Camera size={13} />
              <span>Start Camera</span>
            </button>
          )}
        </div>
      </div>

      {/* Video Viewport Area */}
      <div className="relative w-full aspect-video bg-black flex items-center justify-center overflow-hidden">
        {/* HTML5 Video element */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover scale-x-[-1] ${
            !isCameraActive ? 'hidden' : 'block'
          }`}
        />

        {/* MediaPipe landmark Canvas overlay */}
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full object-cover pointer-events-none scale-x-[-1] ${
            !isCameraActive || !showOverlay ? 'hidden' : 'block'
          }`}
        />

        {/* Fallback & Status Overlays */}
        {!isCameraActive && (
          <div className="flex flex-col items-center justify-center gap-3 p-6 text-center">
            <div className="w-14 h-14 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400">
              <Camera size={26} />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-300 mb-1">
                Camera is currently stopped
              </p>
              <p className="text-xs text-slate-500 max-w-sm">
                Click "Start Camera" above to initiate real-time ASL alphabet recognition. Video frames remain strictly local in your browser.
              </p>
            </div>
            <button
              onClick={onStartCamera}
              className="mt-1 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors shadow-md"
            >
              Enable Camera & Start Recognition
            </button>
          </div>
        )}

        {isCameraActive && detectionState === 'initializing' && (
          <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-xs flex items-center justify-center gap-2 text-cyan-400 text-xs font-mono">
            <RefreshCw className="animate-spin" size={16} />
            <span>Loading MediaPipe hand tracking model...</span>
          </div>
        )}

        {isCameraActive && detectionState === 'permission_denied' && (
          <div className="absolute inset-0 bg-rose-950/90 flex flex-col items-center justify-center gap-2 p-6 text-center text-rose-200">
            <AlertCircle size={32} className="text-rose-400" />
            <p className="text-sm font-bold">Camera Permission Denied</p>
            <p className="text-xs text-rose-300/80 max-w-sm">
              Please click the camera icon in your browser's address bar to grant camera permission for this app.
            </p>
          </div>
        )}

        {/* Live Tracking HUD Banner */}
        {isCameraActive && (
          <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between pointer-events-none">
            <div className="flex items-center gap-2 px-2.5 py-1 bg-black/70 backdrop-blur-md border border-slate-700/60 rounded text-[11px] font-mono">
              {detectionState === 'hand_detected' ? (
                <>
                  <span className="text-emerald-400">● Hand Detected</span>
                  <span className="text-slate-500">·</span>
                  <span className="text-slate-300">21 points</span>
                </>
              ) : (
                <span className="text-amber-400">▲ No hand detected</span>
              )}
            </div>

            <div className="px-2 py-1 bg-black/70 backdrop-blur-md border border-slate-700/60 rounded text-[10px] font-mono text-slate-400">
              Mirrored View
            </div>
          </div>
        )}
      </div>

      {/* Error notification bar if camera fails */}
      {errorMessage && (
        <div className="px-4 py-2 bg-rose-950/60 border-t border-rose-900/60 text-xs text-rose-300 flex items-center gap-2">
          <AlertCircle size={14} className="shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
