"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import PoseCapture, { CapturedPose, PoseKey } from "@/components/PoseCapture";
import VideoCapture360 from "@/components/VideoCapture360";
import MeasurementResults from "@/components/MeasurementResults";

const BodyViewer3D = dynamic(() => import("@/components/BodyViewer3D"), { ssr: false });

type Mode = "standard" | "premium";
type Status = "idle" | "capturing" | "form" | "analyzing" | "done" | "error";

interface SliceData {
  y: number;
  contour: number[][];
}

interface AnalysisResult {
  measurements: Record<string, number>;
  vertices?: number[][];
  faces?: number[][];
  keypoints_3d?: Record<string, number[]>;
  slices?: Record<string, SliceData>;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("standard");
  const [status, setStatus] = useState<Status>("idle");
  const [photos, setPhotos] = useState<Record<PoseKey, CapturedPose> | null>(null);
  const [video, setVideo] = useState<{ file: File; preview: string } | null>(null);
  const [height, setHeight] = useState(170);
  const [weight, setWeight] = useState(70);
  const [gender, setGender] = useState<"homme" | "femme">("femme");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string>("");

  const handlePhotosComplete = (captured: Record<PoseKey, CapturedPose>) => {
    setPhotos(captured);
    setStatus("form");
  };

  const handleVideoComplete = (file: File, preview: string) => {
    setVideo({ file, preview });
    setStatus("form");
  };

  const handleAnalyze = async () => {
    setStatus("analyzing");
    setError("");

    try {
      const formData = new FormData();
      formData.append("mode", mode);
      formData.append("height_cm", height.toString());
      formData.append("weight_kg", weight.toString());
      formData.append("gender", gender);

      if (mode === "standard" && photos) {
        formData.append("photo_front", photos.front.file);
        formData.append("photo_left", photos.left.file);
        formData.append("photo_back", photos.back.file);
        formData.append("photo_right", photos.right.file);
      } else if (mode === "premium" && video) {
        formData.append("video", video.file);
      }

      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Erreur lors de l'analyse");
      }

      const data = await res.json();
      setResult(data);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
      setStatus("error");
    }
  };

  const handleReset = () => {
    setPhotos(null);
    setVideo(null);
    setResult(null);
    setError("");
    setStatus("idle");
  };

  const previewUrl = photos?.front.preview || video?.preview;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="glass sticky top-0 z-50 px-4 py-3">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <h1 className="text-lg font-bold">
            <span className="text-[var(--primary-light)]">ETEKA</span>{" "}
            <span className="text-[var(--foreground)]/60 font-normal">Body Scan</span>
          </h1>
          {status !== "idle" && (
            <button onClick={handleReset} className="text-xs text-[var(--primary-light)] hover:underline">
              Recommencer
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 px-4 py-6 max-w-5xl mx-auto w-full">
        {/* Mode selection */}
        {status === "idle" && (
          <div className="space-y-8">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold">Analysez votre morphologie by ETEKA</h2>
              <p className="text-sm text-[var(--foreground)]/60">
                Precision jusqu&apos;a +/- 2 cm avec reconstruction 3D multi-vue
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-4 max-w-3xl mx-auto">
              <button
                onClick={() => {
                  setMode("standard");
                  setStatus("capturing");
                }}
                className="glass rounded-2xl p-6 text-left hover:bg-[var(--surface-light)] transition-colors"
              >
                <div className="text-3xl mb-2">📸</div>
                <h3 className="font-semibold text-[var(--primary-light)] mb-1">Mode Standard</h3>
                <p className="text-sm text-[var(--foreground)]/60 mb-3">
                  4 photos : face, dos, profil gauche et droit
                </p>
                <div className="text-xs space-y-1 text-[var(--foreground)]/50">
                  <div>⏱ ~60 secondes</div>
                  <div>🎯 Precision +/- 2-3 cm</div>
                  <div>📱 Compatible mobile</div>
                </div>
              </button>

              <button
                onClick={() => {
                  setMode("premium");
                  setStatus("capturing");
                }}
                className="glass rounded-2xl p-6 text-left hover:bg-[var(--surface-light)] transition-colors border-2 border-[var(--accent)]/30"
              >
                <div className="flex items-start justify-between">
                  <div className="text-3xl mb-2">🎥</div>
                  <span className="text-xs bg-[var(--accent)] text-black px-2 py-0.5 rounded-full font-semibold">
                    PREMIUM
                  </span>
                </div>
                <h3 className="font-semibold text-[var(--accent)] mb-1">Scan 360°</h3>
                <p className="text-sm text-[var(--foreground)]/60 mb-3">
                  Video 360° pour une reconstruction ultra-detaillee
                </p>
                <div className="text-xs space-y-1 text-[var(--foreground)]/50">
                  <div>⏱ ~3 minutes</div>
                  <div>🎯 Precision +/- 1 cm</div>
                  <div>✨ Mesh haute definition</div>
                </div>
              </button>
            </div>
          </div>
        )}

        {/* Capture */}
        {status === "capturing" && mode === "standard" && (
          <PoseCapture onComplete={handlePhotosComplete} />
        )}
        {status === "capturing" && mode === "premium" && (
          <VideoCapture360 onComplete={handleVideoComplete} />
        )}

        {/* Form */}
        {status === "form" && (
          <div className="max-w-md mx-auto space-y-5">
            <h3 className="text-lg font-semibold text-center">Informations personnelles</h3>

            <div>
              <label className="block text-sm font-medium mb-2">Genre</label>
              <div className="grid grid-cols-2 gap-2">
                {(["homme", "femme"] as const).map((g) => (
                  <button
                    key={g}
                    onClick={() => setGender(g)}
                    className={`py-2.5 rounded-xl text-sm font-medium transition-all ${
                      gender === g
                        ? "bg-[var(--primary)] text-white"
                        : "bg-[var(--surface)] text-[var(--foreground)]/60 hover:bg-[var(--surface-light)]"
                    }`}
                  >
                    {g.charAt(0).toUpperCase() + g.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Taille (cm)</label>
              <input
                type="number"
                value={height}
                onChange={(e) => setHeight(Number(e.target.value))}
                min={100}
                max={250}
                className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-xl px-4 py-3 focus:outline-none focus:border-[var(--primary)]"
              />
              <input
                type="range"
                value={height}
                onChange={(e) => setHeight(Number(e.target.value))}
                min={100}
                max={250}
                className="w-full mt-2 accent-[var(--primary)]"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Poids (kg)</label>
              <input
                type="number"
                value={weight}
                onChange={(e) => setWeight(Number(e.target.value))}
                min={30}
                max={200}
                className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-xl px-4 py-3 focus:outline-none focus:border-[var(--primary)]"
              />
            </div>

            {error && (
              <div className="bg-[var(--error)]/10 border border-[var(--error)]/30 rounded-xl p-3 text-sm text-[var(--error)]">
                {error}
              </div>
            )}

            <button
              onClick={handleAnalyze}
              className="w-full py-3.5 rounded-xl font-semibold text-white bg-[var(--primary)] hover:bg-[var(--primary-light)] active:scale-[0.98] transition-all"
            >
              Analyser ma morphologie
            </button>
          </div>
        )}

        {/* Analyzing */}
        {status === "analyzing" && (
          <div className="max-w-md mx-auto space-y-4 text-center">
            {previewUrl && (
              <div className="relative w-48 mx-auto rounded-xl overflow-hidden">
                <img src={previewUrl} alt="Analyse" className="w-full object-contain rounded-xl opacity-70" />
                <div className="scan-line absolute left-0 w-full h-0.5 bg-[var(--accent)] shadow-[0_0_12px_var(--accent),0_0_24px_var(--accent)]" />
              </div>
            )}
            <p className="text-sm text-[var(--foreground)]/60">
              Reconstruction 3D multi-vue en cours...
            </p>
            <p className="text-xs text-[var(--foreground)]/40">
              {mode === "premium" ? "Cela peut prendre 2-5 minutes" : "Cela peut prendre 30-60 secondes"}
            </p>
          </div>
        )}

        {/* Results */}
        {status === "done" && result && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Resultats de l&apos;analyse</h2>
              <button onClick={handleReset} className="text-sm text-[var(--primary-light)] hover:underline">
                Nouvelle analyse
              </button>
            </div>

            {result.vertices && result.faces && (
              <BodyViewer3D
                vertices={result.vertices}
                faces={result.faces}
                measurements={result.measurements}
                keypoints={result.keypoints_3d}
                slices={result.slices}
              />
            )}

            <div className="grid md:grid-cols-2 gap-6">
              {previewUrl && (
                <div className="glass rounded-2xl p-4">
                  <img src={previewUrl} alt="Vue de face" className="w-full max-h-[500px] object-contain rounded-xl" />
                  <div className="mt-3 text-center text-sm text-[var(--foreground)]/60">
                    {height} cm / {weight} kg / {gender}
                  </div>
                </div>
              )}
              <MeasurementResults measurements={result.measurements} />
            </div>
          </div>
        )}
      </main>

      <footer className="px-4 py-3 text-center text-xs text-[var(--foreground)]/30">
        ETEKA Body Scan
      </footer>
    </div>
  );
}
