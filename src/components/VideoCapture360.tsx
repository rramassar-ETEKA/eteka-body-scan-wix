"use client";

import { useEffect, useRef, useState } from "react";

interface VideoCapture360Props {
  onComplete: (file: File, preview: string) => void;
}

const TARGET_DURATION = 15; // seconds for full 360

export default function VideoCapture360({ onComplete }: VideoCapture360Props) {
  const [useCamera, setUseCamera] = useState<boolean | null>(null);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string>("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (useCamera && !cameraStream) startCamera();
    return () => {
      cameraStream?.getTracks().forEach((t) => t.stop());
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [useCamera]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      setCameraStream(stream);
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch {
      setCameraError("Impossible d'acceder a la camera. Utilisez l'option upload.");
      setUseCamera(false);
    }
  };

  const startRecording = () => {
    if (!cameraStream) return;

    chunksRef.current = [];
    const recorder = new MediaRecorder(cameraStream, { mimeType: "video/webm" });
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      const file = new File([blob], "scan-360.webm", { type: "video/webm" });
      const preview = URL.createObjectURL(blob);
      onComplete(file, preview);
    };
    recorder.start();
    recorderRef.current = recorder;
    setRecording(true);
    setElapsed(0);

    timerRef.current = setInterval(() => {
      setElapsed((e) => {
        if (e >= TARGET_DURATION) {
          stopRecording();
          return TARGET_DURATION;
        }
        return e + 0.1;
      });
    }, 100);
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    if (timerRef.current) clearInterval(timerRef.current);
    setRecording(false);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const preview = URL.createObjectURL(file);
    onComplete(file, preview);
  };

  if (useCamera === null) {
    return (
      <div className="space-y-4 max-w-md mx-auto">
        <h3 className="text-lg font-semibold text-center">Mode Scan 360° Premium</h3>
        <p className="text-sm text-[var(--foreground)]/60 text-center">
          Demandez a quelqu&apos;un de vous filmer en tournant autour de vous sur 360° en 15 secondes,
          ou posez le telephone et tournez sur vous-meme.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => setUseCamera(true)}
            className="glass rounded-xl p-6 hover:bg-[var(--surface-light)] transition-colors"
          >
            <div className="font-medium">Enregistrer</div>
            <div className="text-xs text-[var(--foreground)]/60 mt-1">Via la camera</div>
          </button>
          <button
            onClick={() => setUseCamera(false)}
            className="glass rounded-xl p-6 hover:bg-[var(--surface-light)] transition-colors"
          >
            <div className="font-medium">Upload</div>
            <div className="text-xs text-[var(--foreground)]/60 mt-1">Video deja enregistree</div>
          </button>
        </div>
      </div>
    );
  }

  const progress = (elapsed / TARGET_DURATION) * 100;

  return (
    <div className="space-y-4 w-full max-w-2xl mx-auto">
      <div className="glass rounded-xl p-4 text-center">
        <h3 className="font-semibold text-[var(--primary-light)]">Scan 360° - Mode Premium</h3>
        <p className="text-sm text-[var(--foreground)]/70 mt-1">
          Tournez lentement sur vous-meme en 15 secondes, bras ecartes
        </p>
      </div>

      {useCamera ? (
        <div className="relative glass rounded-2xl overflow-hidden aspect-[3/4] max-h-[600px] mx-auto">
          {cameraError ? (
            <div className="flex items-center justify-center h-full p-6 text-[var(--error)] text-center">
              {cameraError}
            </div>
          ) : (
            <>
              <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
              {recording && (
                <>
                  <div className="absolute top-4 left-4 right-4">
                    <div className="bg-black/60 rounded-full overflow-hidden">
                      <div
                        className="h-2 bg-[var(--accent)] transition-all"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <div className="text-center text-xs mt-1 text-white">
                      {elapsed.toFixed(1)}s / {TARGET_DURATION}s
                    </div>
                  </div>
                  <div className="absolute top-4 right-4 bg-red-500 text-white text-xs px-2 py-1 rounded-full animate-pulse">
                    REC
                  </div>
                </>
              )}
              <button
                onClick={recording ? stopRecording : startRecording}
                className={`absolute bottom-4 left-1/2 -translate-x-1/2 w-16 h-16 rounded-full border-4 border-white hover:scale-105 active:scale-95 transition-transform ${
                  recording ? "bg-red-500" : "bg-white"
                }`}
                aria-label={recording ? "Arreter" : "Demarrer"}
              >
                {recording && <div className="w-6 h-6 bg-white mx-auto rounded-sm" />}
              </button>
            </>
          )}
        </div>
      ) : (
        <div className="glass rounded-2xl overflow-hidden aspect-[3/4] max-h-[600px] mx-auto flex items-center justify-center">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="bg-[var(--primary)] hover:bg-[var(--primary-light)] text-white px-6 py-3 rounded-xl font-medium"
          >
            Choisir une video
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>
      )}
    </div>
  );
}
