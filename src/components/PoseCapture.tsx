"use client";

import { useEffect, useRef, useState } from "react";

export type PoseKey = "front" | "left" | "back" | "right";

export interface CapturedPose {
  key: PoseKey;
  file: File;
  preview: string;
}

interface PoseCaptureProps {
  onComplete: (photos: Record<PoseKey, CapturedPose>) => void;
}

const POSES: { key: PoseKey; label: string; instruction: string }[] = [
  {
    key: "front",
    label: "Face",
    instruction: "Face a la camera, bras ecartes a 45 degres, pieds legerement ecartes",
  },
  {
    key: "left",
    label: "Profil gauche",
    instruction: "Tournez-vous de 90 degres vers votre droite (votre cote gauche face a la camera)",
  },
  {
    key: "back",
    label: "Dos",
    instruction: "Tournez-vous de dos, bras ecartes a 45 degres",
  },
  {
    key: "right",
    label: "Profil droit",
    instruction: "Tournez-vous de 90 degres vers votre gauche (votre cote droit face a la camera)",
  },
];

function SilhouetteSVG({ pose }: { pose: PoseKey }) {
  if (pose === "front" || pose === "back") {
    return (
      <svg viewBox="0 0 100 200" className="pose-silhouette w-full h-full" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="50" cy="20" r="10" />
        <path d="M 50 30 L 50 100" />
        <path d="M 50 45 L 25 75" />
        <path d="M 50 45 L 75 75" />
        <path d="M 50 100 L 40 170" />
        <path d="M 50 100 L 60 170" />
        <path d="M 40 170 L 38 190" />
        <path d="M 60 170 L 62 190" />
        <path d="M 40 60 Q 50 70 60 60" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 100 200" className="pose-silhouette w-full h-full" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="50" cy="20" r="10" />
      <path d="M 50 30 L 50 100" />
      <path d="M 50 50 L 40 90" />
      <path d="M 50 100 L 45 170" />
      <path d="M 45 170 L 48 190" />
    </svg>
  );
}

export default function PoseCapture({ onComplete }: PoseCaptureProps) {
  const [currentPoseIdx, setCurrentPoseIdx] = useState(0);
  const [captured, setCaptured] = useState<Partial<Record<PoseKey, CapturedPose>>>({});
  const [useCamera, setUseCamera] = useState<boolean | null>(null);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string>("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const currentPose = POSES[currentPoseIdx];
  const allCaptured = POSES.every((p) => captured[p.key]);

  useEffect(() => {
    if (useCamera && !cameraStream) {
      startCamera();
    }
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [useCamera]);

  useEffect(() => {
    if (allCaptured) {
      onComplete(captured as Record<PoseKey, CapturedPose>);
    }
  }, [allCaptured, captured, onComplete]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      setCameraError("Impossible d'acceder a la camera. Utilisez l'option upload.");
      setUseCamera(false);
    }
  };

  const captureFromCamera = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], `${currentPose.key}.jpg`, { type: "image/jpeg" });
      const preview = URL.createObjectURL(blob);
      addCapture({ key: currentPose.key, file, preview });
    }, "image/jpeg", 0.92);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const preview = URL.createObjectURL(file);
    addCapture({ key: currentPose.key, file, preview });
  };

  const addCapture = (pose: CapturedPose) => {
    const next = { ...captured, [pose.key]: pose };
    setCaptured(next);
    // Advance to next uncaptured pose
    const nextIdx = POSES.findIndex((p) => !next[p.key]);
    if (nextIdx >= 0) {
      setCurrentPoseIdx(nextIdx);
    }
  };

  const retake = (key: PoseKey) => {
    const next = { ...captured };
    delete next[key];
    setCaptured(next);
    setCurrentPoseIdx(POSES.findIndex((p) => p.key === key));
  };

  if (useCamera === null) {
    return (
      <div className="space-y-4 max-w-md mx-auto">
        <h3 className="text-lg font-semibold text-center">Comment souhaitez-vous prendre les photos ?</h3>
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => setUseCamera(true)}
            className="glass rounded-xl p-6 hover:bg-[var(--surface-light)] transition-colors"
          >
            <div className="font-medium">Camera</div>
            <div className="text-xs text-[var(--foreground)]/60 mt-1">Directement via l&apos;appareil</div>
          </button>
          <button
            onClick={() => setUseCamera(false)}
            className="glass rounded-xl p-6 hover:bg-[var(--surface-light)] transition-colors"
          >
            <div className="font-medium">Upload</div>
            <div className="text-xs text-[var(--foreground)]/60 mt-1">Photos deja prises</div>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 w-full max-w-2xl mx-auto">
      {/* Progress indicator */}
      <div className="flex items-center justify-between">
        {POSES.map((pose, idx) => (
          <div
            key={pose.key}
            className="flex-1 flex flex-col items-center cursor-pointer"
            onClick={() => captured[pose.key] && retake(pose.key)}
          >
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all ${
                captured[pose.key]
                  ? "bg-[var(--success)] border-[var(--success)] text-white"
                  : idx === currentPoseIdx
                  ? "border-[var(--primary)] text-[var(--primary-light)]"
                  : "border-[var(--border)] text-[var(--foreground)]/40"
              }`}
            >
              {captured[pose.key] ? "✓" : idx + 1}
            </div>
            <span className="text-xs mt-1 text-[var(--foreground)]/60">{pose.label}</span>
          </div>
        ))}
      </div>

      {/* Current pose instruction */}
      <div className="glass rounded-xl p-4 text-center">
        <h3 className="font-semibold text-[var(--primary-light)]">Pose {currentPoseIdx + 1}/4 : {currentPose.label}</h3>
        <p className="text-sm text-[var(--foreground)]/70 mt-1">{currentPose.instruction}</p>
      </div>

      {/* Camera or upload view */}
      {useCamera ? (
        <div className="relative glass rounded-2xl overflow-hidden aspect-[3/4] max-h-[600px] mx-auto">
          {cameraError ? (
            <div className="flex items-center justify-center h-full p-6 text-[var(--error)] text-center">
              {cameraError}
            </div>
          ) : (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              {/* Silhouette overlay */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-[var(--accent)]">
                <div className="h-[85%]">
                  <SilhouetteSVG pose={currentPose.key} />
                </div>
              </div>
              {/* Capture button */}
              <button
                onClick={captureFromCamera}
                className="absolute bottom-4 left-1/2 -translate-x-1/2 w-16 h-16 rounded-full bg-white border-4 border-[var(--primary)] hover:scale-105 active:scale-95 transition-transform"
                aria-label="Capturer"
              />
            </>
          )}
          <canvas ref={canvasRef} className="hidden" />
        </div>
      ) : (
        <div className="relative glass rounded-2xl overflow-hidden aspect-[3/4] max-h-[600px] mx-auto flex items-center justify-center">
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-[var(--accent)]">
            <div className="h-[85%]">
              <SilhouetteSVG pose={currentPose.key} />
            </div>
          </div>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="relative z-10 bg-[var(--primary)] hover:bg-[var(--primary-light)] text-white px-6 py-3 rounded-xl font-medium"
          >
            Choisir une photo
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>
      )}

      {/* Captured thumbnails */}
      {Object.keys(captured).length > 0 && (
        <div className="grid grid-cols-4 gap-2">
          {POSES.map((pose) => {
            const cap = captured[pose.key];
            return (
              <div key={pose.key} className="relative">
                {cap ? (
                  <>
                    <img src={cap.preview} alt={pose.label} className="w-full aspect-[3/4] object-cover rounded-lg" />
                    <button
                      onClick={() => retake(pose.key)}
                      className="absolute top-1 right-1 bg-black/60 text-white text-xs px-2 py-1 rounded hover:bg-black/80"
                    >
                      Refaire
                    </button>
                  </>
                ) : (
                  <div className="w-full aspect-[3/4] rounded-lg border-2 border-dashed border-[var(--border)] flex items-center justify-center text-xs text-[var(--foreground)]/30">
                    {pose.label}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
