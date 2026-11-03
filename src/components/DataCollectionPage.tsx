import React, { useCallback, useMemo, useRef, useState } from 'react';
import { ArrowLeft, Camera, CameraOff, Download, Hand, Play, Square, Trash2 } from 'lucide-react';
import { useHandLandmarker } from '../hooks/useHandLandmarker';
import { LandmarkPoint } from '../types/prediction';

const LETTERS = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'];
const COORD_COLUMNS = Array.from({ length: 21 }, (_, i) => [`x${i}`, `y${i}`, `z${i}`]).flat();

type SampleRow = {
  label: string;
  source: string;
  participant_id: string;
  session_id: string;
  condition: string;
  handedness: string;
  captured_at: string;
  landmarks: number[];
};

function csvCell(value: string | number) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export default function DataCollectionPage() {
  const [letter, setLetter] = useState('A');
  const [participant, setParticipant] = useState('participant_01');
  const [session, setSession] = useState(() => `session_${new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}`);
  const [condition, setCondition] = useState('normal_lighting_front_view');
  const [isCapturing, setIsCapturing] = useState(false);
  const [samples, setSamples] = useState<SampleRow[]>([]);
  const [lastHand, setLastHand] = useState('Not detected');
  const [lastCaptureAt, setLastCaptureAt] = useState<number>(0);
  const [notice, setNotice] = useState('Start the camera, select a letter, then start capture.');
  const samplesRef = useRef<SampleRow[]>([]);
  const lastCaptureRef = useRef(0);
  const isCapturingRef = useRef(false);
  const labelRef = useRef(letter);
  const participantRef = useRef(participant);
  const sessionRef = useRef(session);
  const conditionRef = useRef(condition);
  labelRef.current = letter;
  participantRef.current = participant;
  sessionRef.current = session;
  conditionRef.current = condition;
  isCapturingRef.current = isCapturing;

  const handleLandmarks = useCallback((landmarks: LandmarkPoint[], _mediaPipeMs: number, handedness = 'Unknown') => {
    setLastHand(handedness);
    if (!isCapturingRef.current) return;
    const now = Date.now();
    // 3 samples/second avoids saving every near-identical camera frame.
    if (now - lastCaptureRef.current < 330) return;
    if (landmarks.length !== 21) return;
    const flat = landmarks.flatMap((point) => [point.x, point.y, point.z]);
    if (flat.length !== 63 || !flat.every(Number.isFinite)) return;
    // Reject degenerate detections and implausible coordinates before saving.
    const wrist = [flat[0], flat[1], flat[2]];
    const maxDistance = Math.max(...Array.from({ length: 21 }, (_, i) => {
      const dx = flat[i * 3] - wrist[0];
      const dy = flat[i * 3 + 1] - wrist[1];
      const dz = flat[i * 3 + 2] - wrist[2];
      return Math.hypot(dx, dy, dz);
    }));
    if (maxDistance < 1e-4 || maxDistance > 5) return;
    const row: SampleRow = {
      label: labelRef.current,
      source: 'webcam',
      participant_id: participantRef.current.trim(),
      session_id: sessionRef.current.trim(),
      condition: conditionRef.current,
      handedness,
      captured_at: new Date(now).toISOString(),
      landmarks: flat,
    };
    samplesRef.current.push(row);
    lastCaptureRef.current = now;
    setSamples([...samplesRef.current]);
    setLastCaptureAt(now);
  }, []);

  const { videoRef, canvasRef, detectionState, errorMessage, isCameraActive, fps, startCamera, stopCamera } = useHandLandmarker(handleLandmarks);
  const countForLetter = samples.filter((sample) => sample.label === letter).length;
  const counts = useMemo(() => LETTERS.map((item) => ({ letter: item, count: samples.filter((s) => s.label === item).length })), [samples]);
  const totalTarget = 300;

  const startCapture = () => {
    if (!participant.trim() || !session.trim()) {
      setNotice('Participant ID and session ID are required.');
      return;
    }
    if (!isCameraActive) {
      setNotice('Start the camera before collecting samples.');
      return;
    }
    lastCaptureRef.current = 0;
    setIsCapturing(true);
    setNotice(`Collecting ${letter}. Hold the sign, then vary its angle and distance slowly.`);
  };

  const stopCapture = () => {
    setIsCapturing(false);
    setNotice(`Capture paused. ${samplesRef.current.length} total samples are in this session.`);
  };

  const downloadCsv = () => {
    const exportSamples = samplesRef.current.filter((sample) => sample.session_id === sessionRef.current);
    if (!exportSamples.length) return;
    const header = ['label', 'source', 'participant_id', 'session_id', 'condition', 'handedness', 'captured_at', ...COORD_COLUMNS];
    const lines = [header.join(',')];
    for (const sample of exportSamples) {
      lines.push([
        sample.label, sample.source, sample.participant_id, sample.session_id, sample.condition,
        sample.handedness, sample.captured_at, ...sample.landmarks,
      ].map(csvCell).join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `signal_landmarks_${participant}_${session}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
    setNotice(`Exported ${exportSamples.length} raw landmark samples from ${sessionRef.current}. Keep this CSV in data/raw/.`);
  };

  const stopCameraSafely = () => {
    setIsCapturing(false);
    stopCamera();
  };

  const startNewSession = () => {
    if (isCapturing) return;
    setSession(`session_${new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}`);
    setNotice('New recording session created. Collect all 24 letters under this session ID before starting another session.');
  };

  const clearSamples = () => {
    if (!window.confirm('Delete all samples collected in this browser session? This cannot be undone.')) return;
    samplesRef.current = [];
    setSamples([]);
    setNotice('Local collection buffer cleared.');
  };

  return (
    <main className="min-h-screen bg-[#0b0f17] text-slate-100 px-4 py-6 md:px-8">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div>
            <a href="/" className="inline-flex items-center gap-2 text-sm text-cyan-300 hover:text-cyan-200 mb-3"><ArrowLeft size={15}/> Back to SIGNAL</a>
            <p className="text-xs tracking-[0.25em] uppercase text-cyan-400 font-semibold">SIGNAL / V2 DATA LAB</p>
            <h1 className="text-3xl font-bold mt-1">Real-world landmark collection</h1>
            <p className="text-sm text-slate-400 mt-2">Capture raw 21-point landmarks from the same MediaPipe Tasks Vision pipeline used by the live app.</p>
          </div>
          <div className="rounded-xl border border-cyan-900 bg-cyan-950/30 px-4 py-3 text-sm text-cyan-200">Local-only capture · No video is uploaded</div>
        </header>

        <div className="grid lg:grid-cols-[1.35fr_0.9fr] gap-6">
          <section className="rounded-2xl overflow-hidden border border-slate-800 bg-[#111622]">
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
              <div className="text-xs font-mono uppercase tracking-wider text-slate-300">Webcam / MediaPipe stream</div>
              <div className="text-xs text-slate-400">{isCameraActive ? `${fps} FPS` : 'Camera off'} · {detectionState}</div>
            </div>
            <div className="relative aspect-video bg-black">
              <video ref={videoRef} autoPlay playsInline muted className={`w-full h-full object-cover scale-x-[-1] ${isCameraActive ? 'block' : 'hidden'}`} />
              <canvas ref={canvasRef} className={`absolute inset-0 w-full h-full object-cover pointer-events-none scale-x-[-1] ${isCameraActive ? 'block' : 'hidden'}`} />
              {!isCameraActive && <div className="absolute inset-0 flex flex-col items-center justify-center text-center text-slate-500 gap-2"><Camera size={36}/><p>Start your camera to begin landmark collection.</p></div>}
              <div className="absolute bottom-3 left-3 rounded-lg bg-black/70 px-3 py-2 text-xs font-mono">{detectionState === 'hand_detected' ? `21 landmarks · ${lastHand} hand` : 'Waiting for hand detection'}</div>
              {isCapturing && <div className="absolute top-3 right-3 rounded-lg bg-rose-600/90 px-3 py-2 text-xs font-bold animate-pulse">● RECORDING</div>}
            </div>
            <div className="flex flex-wrap gap-2 p-4">
              {!isCameraActive ? <button onClick={startCamera} className="rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 inline-flex items-center gap-2"><Camera size={16}/> Start camera</button> : <button onClick={stopCameraSafely} className="rounded-lg bg-slate-700 px-4 py-2 text-sm font-semibold inline-flex items-center gap-2"><CameraOff size={16}/> Stop camera</button>}
              {!isCapturing ? <button onClick={startCapture} disabled={!isCameraActive} className="rounded-lg bg-emerald-500 disabled:opacity-40 px-4 py-2 text-sm font-semibold text-slate-950 inline-flex items-center gap-2"><Play size={16}/> Start capture</button> : <button onClick={stopCapture} className="rounded-lg bg-rose-500 px-4 py-2 text-sm font-semibold text-white inline-flex items-center gap-2"><Square size={15}/> Pause capture</button>}
              <button onClick={downloadCsv} disabled={!samples.length} className="rounded-lg border border-cyan-700 px-4 py-2 text-sm font-semibold text-cyan-200 disabled:opacity-40 inline-flex items-center gap-2"><Download size={16}/> Export CSV</button>
              <button onClick={clearSamples} disabled={!samples.length || isCapturing} className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 disabled:opacity-40 inline-flex items-center gap-2"><Trash2 size={15}/> Clear</button>
            </div>
          </section>

          <section className="rounded-2xl border border-slate-800 bg-[#111622] p-5 space-y-4">
            <div><h2 className="font-semibold text-lg">Collection metadata</h2><p className="text-xs text-slate-500 mt-1">Use the same participant ID across your recordings and a new session ID for each separate recording session.</p></div>
            <label className="block text-xs text-slate-400">Participant ID<input value={participant} onChange={(e) => setParticipant(e.target.value)} disabled={isCapturing} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white disabled:opacity-50" /></label>
            <label className="block text-xs text-slate-400">Session ID<input value={session} onChange={(e) => setSession(e.target.value)} disabled={isCapturing} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white" /></label><button onClick={startNewSession} disabled={isCapturing} className="w-full rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 disabled:opacity-40">Start a new recording session</button>
            <label className="block text-xs text-slate-400">Recording condition<select value={condition} onChange={(e) => setCondition(e.target.value)} disabled={isCapturing} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white disabled:opacity-50"><option value="normal_lighting_front_view">Normal lighting · front view</option><option value="bright_lighting">Bright lighting</option><option value="low_lighting">Low lighting</option><option value="left_tilt">Hand tilted left</option><option value="right_tilt">Hand tilted right</option><option value="closer_camera">Closer to camera</option><option value="farther_camera">Farther from camera</option><option value="busy_background">Busy background</option></select></label>
            <div className="rounded-xl bg-slate-950 p-4 space-y-2">
              <div className="text-xs text-slate-400">Current sign label</div>
              <div className="flex items-center justify-between"><button onClick={() => setLetter(LETTERS[Math.max(0, LETTERS.indexOf(letter)-1)])} disabled={isCapturing} className="rounded bg-slate-800 px-3 py-2 disabled:opacity-40">←</button><div className="text-5xl font-bold text-cyan-300">{letter}</div><button onClick={() => setLetter(LETTERS[Math.min(LETTERS.length-1, LETTERS.indexOf(letter)+1)])} disabled={isCapturing} className="rounded bg-slate-800 px-3 py-2 disabled:opacity-40">→</button></div>
              <select value={letter} onChange={(e) => setLetter(e.target.value)} disabled={isCapturing} className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"><option value="">Choose a letter</option>{LETTERS.map((item) => <option key={item} value={item}>{item}</option>)}</select>
              <div className="flex justify-between text-xs text-slate-400"><span>Samples for {letter}</span><span>{countForLetter} / {totalTarget} starter target</span></div>
              <div className="h-2 rounded-full bg-slate-800 overflow-hidden"><div className="h-full bg-cyan-400" style={{ width: `${Math.min(100, countForLetter / totalTarget * 100)}%` }}/></div>
            </div>
            <div className="text-xs text-slate-400 flex items-center gap-2"><Hand size={15}/> Last detected hand: <span className="text-slate-200">{lastHand}</span></div>
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3 text-xs text-slate-400">{notice}{lastCaptureAt > 0 && <span className="block mt-1">Last sample: {new Date(lastCaptureAt).toLocaleTimeString()}</span>}</div>
            {errorMessage && <p className="text-sm text-rose-300">{errorMessage}</p>}
          </section>
        </div>

        <section className="rounded-2xl border border-slate-800 bg-[#111622] p-5">
          <div className="flex flex-wrap justify-between gap-2 items-center mb-4"><div><h2 className="font-semibold">Dataset coverage</h2><p className="text-xs text-slate-500">Aim for 300–500 varied samples per letter, across multiple separate sessions.</p></div><div className="text-sm text-slate-300">{samples.length.toLocaleString()} total samples</div></div>
          <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-12 gap-2">
            {counts.map(({ letter: item, count }) => <button key={item} onClick={() => !isCapturing && setLetter(item)} disabled={isCapturing} className={`rounded-lg border p-2 text-left disabled:opacity-60 ${item === letter ? 'border-cyan-400 bg-cyan-950/40' : 'border-slate-800 bg-slate-950'}`}><div className="font-bold text-lg">{item}</div><div className="text-[10px] text-slate-500">{count}</div><div className="mt-1 h-1 rounded bg-slate-800 overflow-hidden"><div className="h-full bg-emerald-400" style={{ width: `${Math.min(100, count / totalTarget * 100)}%` }}/></div></button>)}
          </div>
          <p className="mt-4 text-xs text-amber-300">Collection guidance: record separate takes. Between takes, lower your hand, reset the pose, then sign again. Vary distance, tilt and lighting. Do not treat adjacent video frames as independent examples.</p>
        </section>
      </div>
    </main>
  );
}
