/**
 * Custom React hook for MediaPipe HandLandmarker lifecycle and real-time detection.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { LandmarkPoint, HandDetectionState } from '../types/prediction';

// MediaPipe 21 Hand Landmarks connection topology
export const HAND_CONNECTIONS: [number, number][] = [
  // Wrist to MCPs
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17]                               // Palm base connection
];

export function useHandLandmarker(
  onLandmarksDetected?: (
    landmarks: LandmarkPoint[],
    mediaPipeMs: number,
    handedness?: string,
  ) => void,
) {
  const [detectionState, setDetectionState] = useState<HandDetectionState>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);
  const frameCounterRef = useRef<number>(0);
  const lastFpsCalcTimeRef = useRef<number>(performance.now());
  const onLandmarksDetectedRef = useRef(onLandmarksDetected);
  onLandmarksDetectedRef.current = onLandmarksDetected;

  // Initialize MediaPipe HandLandmarker
  useEffect(() => {
    let isCancelled = false;

    async function initLandmarker() {
      try {
        setDetectionState('initializing');
        
        // Use local WASM files with CDN fallback
        let vision;
        try {
          vision = await FilesetResolver.forVisionTasks('/wasm');
        } catch (e) {
          console.warn('[MediaPipe] Local WASM load failed, falling back to CDN:', e);
          vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
          );
        }

        if (isCancelled) return;

        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/models/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.65,
          minHandPresenceConfidence: 0.65,
          minTrackingConfidence: 0.6,
        });

        if (isCancelled) {
          landmarker.close();
          return;
        }

        handLandmarkerRef.current = landmarker;
        setDetectionState('camera_ready');
      } catch (err: any) {
        if (!isCancelled) {
          console.error('[MediaPipe] Failed to initialize hand landmarker:', err);
          setErrorMessage(err?.message || 'Failed to initialize vision tracking model');
          setDetectionState('camera_error');
        }
      }
    }

    initLandmarker();

    return () => {
      isCancelled = true;
      if (handLandmarkerRef.current) {
        try {
          handLandmarkerRef.current.close();
        } catch {}
      }
    };
  }, []);

  // Frame processing loop
  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = handLandmarkerRef.current;

    if (!video || !canvas || !landmarker || video.readyState < 2) {
      animationFrameIdRef.current = requestAnimationFrame(processFrame);
      return;
    }

    // Update FPS metric every 500ms
    frameCounterRef.current++;
    const now = performance.now();
    if (now - lastFpsCalcTimeRef.current >= 500) {
      const computedFps = Math.round(
        (frameCounterRef.current * 1000) / (now - lastFpsCalcTimeRef.current)
      );
      setFps(computedFps);
      frameCounterRef.current = 0;
      lastFpsCalcTimeRef.current = now;
    }

    // Only process if video time has advanced
    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;

      // Sync canvas dimensions to video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const mediaPipeStart = performance.now();
        const results = landmarker.detectForVideo(video, now);
        const mediaPipeMs = performance.now() - mediaPipeStart;

        if (results && results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0]; // Primary detected hand
          setDetectionState('hand_detected');

          // Draw hand connections (white/cyan skeleton)
          ctx.lineWidth = 2.5;
          ctx.strokeStyle = '#06B6D4'; // Laser cyan skeleton
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';

          for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
            const p1 = landmarks[startIdx];
            const p2 = landmarks[endIdx];
            ctx.beginPath();
            ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
            ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
            ctx.stroke();
          }

          // Draw hand landmarks (nodes)
          for (let i = 0; i < landmarks.length; i++) {
            const lm = landmarks[i];
            const px = lm.x * canvas.width;
            const py = lm.y * canvas.height;

            // Draw outer glow/ring
            ctx.beginPath();
            ctx.arc(px, py, i === 0 ? 6 : 4.5, 0, 2 * Math.PI);
            ctx.fillStyle = i === 0 ? '#38BDF8' : '#EF4444'; // Wrist is blue, fingers are bright red
            ctx.fill();

            // Inner white dot
            ctx.beginPath();
            ctx.arc(px, py, 1.8, 0, 2 * Math.PI);
            ctx.fillStyle = '#FFFFFF';
            ctx.fill();
          }

          // Notify parent callback
          if (onLandmarksDetectedRef.current) {
            const handedness = results.handednesses?.[0]?.[0]?.categoryName ?? 'Unknown';
            onLandmarksDetectedRef.current(landmarks, mediaPipeMs, handedness);
          }
        } else {
          setDetectionState('no_hand');
        }
      }
    }

    animationFrameIdRef.current = requestAnimationFrame(processFrame);
  }, []);

  // Start webcam
  const startCamera = useCallback(async () => {
    try {
      setErrorMessage(null);

      // Verify camera support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Webcam mediaDevices API is not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().then(() => {
            setIsCameraActive(true);
            setDetectionState('detecting');
            animationFrameIdRef.current = requestAnimationFrame(processFrame);
          });
        };
      }
    } catch (err: any) {
      console.error('[Webcam] Access error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setDetectionState('permission_denied');
        setErrorMessage('Camera access was denied. Please allow camera permissions in your browser.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setDetectionState('camera_error');
        setErrorMessage('No camera hardware found on this device.');
      } else {
        setDetectionState('camera_error');
        setErrorMessage(err?.message || 'Unable to start camera.');
      }
      setIsCameraActive(false);
    }
  }, [processFrame]);

  // Stop webcam
  const stopCamera = useCallback(() => {
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }

    setIsCameraActive(false);
    setFps(0);
    setDetectionState('camera_ready');
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return {
    videoRef,
    canvasRef,
    detectionState,
    errorMessage,
    isCameraActive,
    fps,
    startCamera,
    stopCamera,
  };
}
