"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import PoseCapture, { CapturedPose, PoseKey } from "@/components/PoseCapture";
import VideoCapture360 from "@/components/VideoCapture360";
import MeasurementResults from "@/components/MeasurementResults";
import CaptureInstructions from "@/components/CaptureInstructions";

const BodyViewer3D = dynamic(() => import("@/components/BodyViewer3D"), { ssr: false });

type Mode = "standard" | "premium";
type Status = "idle" | "instructions" | "capturing" | "form" | "analyzing" | "done" | "error";

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
  uvs?: number[][];
  texture_b64?: string;
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
  const [embedMode, setEmbedMode] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [extractPct, setExtractPct] = useState(0);
  const [pollSec, setPollSec] = useState(0);
  const [pollCount, setPollCount] = useState(0);
  const [pollStatus, setPollStatus] = useState<string>("-");
  const [pollJobId, setPollJobId] = useState<string>("");
  const [pollModalUrl, setPollModalUrl] = useState<string>("");
  const [analyzePhase, setAnalyzePhase] = useState<"extract" | "upload" | "processing">("upload");
  const mainRef = useRef<HTMLDivElement>(null);

  // Detect embed mode from URL param
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const embed = params.get("embed") === "1";
    setEmbedMode(embed);
    if (embed) document.body.classList.add("embed");
  }, []);

  // PostMessage to parent (Wix iframe) to resize iframe height dynamically
  useEffect(() => {
    if (!embedMode || typeof window === "undefined") return;
    const sendHeight = () => {
      const h = document.documentElement.scrollHeight;
      window.parent.postMessage({ type: "eteka-bodyscan-resize", height: h }, "*");
    };
    sendHeight();
    const observer = new ResizeObserver(sendHeight);
    if (mainRef.current) observer.observe(mainRef.current);
    return () => observer.disconnect();
  }, [embedMode, status, result]);

  const handlePhotosComplete = (captured: Record<PoseKey, CapturedPose>) => {
    setPhotos(captured);
    setStatus("form");
  };

  const handleVideoComplete = (file: File, preview: string) => {
    setVideo({ file, preview });
    setStatus("form");
  };

  const extractVideoFrames = async (
    file: File,
    count = 32,
    onProgress?: (i: number, total: number) => void,
    maxDim = 1280,
    quality = 0.85,
  ): Promise<File[]> => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.src = url;
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("Lecture video impossible"));
    });

    // WebM from MediaRecorder often has Infinity duration; force a metadata refresh.
    if (!isFinite(video.duration) || video.duration === 0) {
      await new Promise<void>((resolve) => {
        const onUpdate = () => {
          if (isFinite(video.duration) && video.duration > 0) {
            video.removeEventListener("timeupdate", onUpdate);
            video.currentTime = 0;
            resolve();
          }
        };
        video.addEventListener("timeupdate", onUpdate);
        video.currentTime = 1e9;
        setTimeout(resolve, 3000);
      });
    }

    const duration = video.duration;
    if (!duration || !isFinite(duration) || duration < 0.5) {
      URL.revokeObjectURL(url);
      throw new Error("Video invalide ou trop courte");
    }

    const W = video.videoWidth;
    const H = video.videoHeight;
    if (!W || !H) {
      URL.revokeObjectURL(url);
      throw new Error("Resolution video invalide");
    }
    const scale = Math.min(1, maxDim / Math.max(W, H));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(W * scale);
    canvas.height = Math.round(H * scale);
    const ctx = canvas.getContext("2d")!;

    const frames: File[] = [];
    for (let i = 0; i < count; i++) {
      const t = (i / Math.max(1, count - 1)) * duration * 0.999;
      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        video.addEventListener("seeked", onSeeked);
        video.currentTime = t;
        setTimeout(() => {
          video.removeEventListener("seeked", onSeeked);
          resolve();
        }, 3000);
      });
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", quality);
      });
      if (blob) frames.push(new File([blob], `frame_${i}.jpg`, { type: "image/jpeg" }));
      onProgress?.(i + 1, count);
    }

    URL.revokeObjectURL(url);
    return frames;
  };

  const compressImage = async (file: File, maxDim = 1280, quality = 0.85): Promise<File> => {
    // createImageBitmap with imageOrientation respects EXIF rotation (avoids sideways photos from phones)
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (!blob) return reject(new Error("Compression failed"));
          resolve(new File([blob], file.name.replace(/\.[^.]+$/, ".jpg"), { type: "image/jpeg" }));
        },
        "image/jpeg",
        quality
      );
    });
  };

  const pollOnce = async (modalUrl: string, jobId: string): Promise<{ status: string; result?: AnalysisResult; detail?: string }> => {
    const resp = await fetch(`${modalUrl}/job/${jobId}`, { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  };

  const pollJob = async (
    modalUrl: string,
    jobId: string,
    onElapsed?: (sec: number) => void,
  ): Promise<AnalysisResult> => {
    const POLL_INTERVAL_MS = 3000;
    const MAX_POLL_MS = 15 * 60 * 1000;
    const t0 = Date.now();
    let count = 0;
    setPollCount(0);
    setPollStatus("starting");
    while (true) {
      const elapsedMs = Date.now() - t0;
      if (elapsedMs > MAX_POLL_MS) {
        throw new Error("Timeout (>15 min) - le serveur met trop longtemps");
      }
      onElapsed?.(Math.round(elapsedMs / 1000));
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      count++;
      setPollCount(count);
      try {
        const pd = await pollOnce(modalUrl, jobId);
        const elapsedSec = Math.round((Date.now() - t0) / 1000);
        console.log(`[Body Scan] Poll ${count} @ ${elapsedSec}s -> ${pd.status}`);
        setPollStatus(pd.status);
        if (pd.status === "done" && pd.result) return pd.result;
        if (pd.status === "error") throw new Error(pd.detail || "Erreur backend");
        if (pd.status === "expired") throw new Error("Resultat expire");
      } catch (e) {
        if (e instanceof Error && (e.message.startsWith("Erreur backend") || e.message.startsWith("Resultat expire") || e.message.startsWith("Timeout"))) {
          throw e;
        }
        const msg = e instanceof Error ? e.message : String(e);
        console.warn(`[Body Scan] Poll ${count} fail (will retry):`, msg);
        setPollStatus(`error retry: ${msg.slice(0, 40)}`);
      }
    }
  };

  const manualCheck = async () => {
    if (!pollJobId || !pollModalUrl) return;
    setPollStatus("checking...");
    try {
      const pd = await pollOnce(pollModalUrl, pollJobId);
      setPollStatus(pd.status);
      if (pd.status === "done" && pd.result) {
        setResult(pd.result);
        setStatus("done");
        try { sessionStorage.removeItem("eteka_pending_job"); } catch {}
      } else if (pd.status === "error") {
        setError(pd.detail || "Erreur backend");
        setStatus("error");
      }
    } catch (e) {
      setPollStatus(`erreur: ${e instanceof Error ? e.message : "reseau"}`);
    }
  };

  // Resume polling on page reload if a job is pending in sessionStorage
  useEffect(() => {
    if (typeof window === "undefined") return;
    let pending: { jobId: string; modalUrl: string; ts: number } | null = null;
    try {
      const raw = sessionStorage.getItem("eteka_pending_job");
      if (raw) pending = JSON.parse(raw);
    } catch {}
    if (!pending || !pending.jobId) return;
    if (Date.now() - pending.ts > 15 * 60 * 1000) {
      sessionStorage.removeItem("eteka_pending_job");
      return;
    }
    console.log(`[Body Scan] Resuming poll for job ${pending.jobId} (started ${Math.round((Date.now() - pending.ts) / 1000)}s ago)`);
    setStatus("analyzing");
    setMode("premium");
    setAnalyzePhase("processing");
    setPollJobId(pending.jobId);
    setPollModalUrl(pending.modalUrl);
    pollJob(pending.modalUrl, pending.jobId, setPollSec)
      .then((d: AnalysisResult) => {
        setResult(d);
        setStatus("done");
        try { sessionStorage.removeItem("eteka_pending_job"); } catch {}
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Erreur inconnue");
        setStatus("error");
        try { sessionStorage.removeItem("eteka_pending_job"); } catch {}
      });
  }, []);

  const handleAnalyze = async () => {
    setStatus("analyzing");
    setError("");
    setUploadPct(0);
    setExtractPct(0);
    setAnalyzePhase(mode === "premium" ? "extract" : "upload");

    const MODAL_URL = process.env.NEXT_PUBLIC_MODAL_API_URL;
    console.log("[Body Scan] handleAnalyze start", { mode, MODAL_URL });
    if (!MODAL_URL) {
      setError("Configuration manquante: NEXT_PUBLIC_MODAL_API_URL");
      setStatus("error");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("height_cm", height.toString());

      let endpoint = "";
      let uploadBytes = 0;

      if (mode === "standard" && photos) {
        endpoint = `${MODAL_URL}/analyze_multiview`;
        const compressed = {
          front: await compressImage(photos.front.file),
          left: await compressImage(photos.left.file),
          back: await compressImage(photos.back.file),
          right: await compressImage(photos.right.file),
        };
        formData.append("photo_front", compressed.front);
        formData.append("photo_left", compressed.left);
        formData.append("photo_back", compressed.back);
        formData.append("photo_right", compressed.right);
        uploadBytes = compressed.front.size + compressed.left.size + compressed.back.size + compressed.right.size;
      } else if (mode === "premium" && video) {
        endpoint = `${MODAL_URL}/analyze_video_frames`;
        const N_FRAMES = 32;
        console.log(`[Body Scan] Extracting ${N_FRAMES} frames from video (${(video.file.size / (1024 * 1024)).toFixed(1)} MB source)...`);
        const t_extract = performance.now();
        const frames = await extractVideoFrames(video.file, N_FRAMES, (i, total) => {
          setExtractPct(Math.round((i / total) * 100));
          if (i % 8 === 0) {
            console.log(`[Body Scan] Extracted ${i}/${total} frames`);
          }
        });
        console.log(`[Body Scan] Extracted ${frames.length} frames in ${((performance.now() - t_extract) / 1000).toFixed(1)}s`);
        for (const f of frames) {
          formData.append("frames", f, f.name);
          uploadBytes += f.size;
        }
        setAnalyzePhase("upload");
      } else {
        throw new Error("Donnees manquantes");
      }

      console.log("[Body Scan] Uploading", {
        endpoint,
        sizeMB: (uploadBytes / (1024 * 1024)).toFixed(2),
      });
      const t0 = performance.now();

      // Upload via XHR to get progress events
      const uploadResponseText: string = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", endpoint);
        xhr.timeout = 5 * 60 * 1000;
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            setUploadPct(pct);
            if (pct % 10 === 0) {
              console.log(`[Body Scan] Upload ${pct}% (${(e.loaded / (1024 * 1024)).toFixed(1)}/${(e.total / (1024 * 1024)).toFixed(1)} MB)`);
            }
          }
        };
        xhr.upload.onload = () => {
          console.log(`[Body Scan] Upload finished in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
          setAnalyzePhase("processing");
        };
        xhr.onerror = () => reject(new Error("Erreur reseau pendant l'upload"));
        xhr.ontimeout = () => reject(new Error("Timeout upload (>5 min)"));
        xhr.onload = () => {
          console.log(`[Body Scan] Upload response status ${xhr.status} after ${((performance.now() - t0) / 1000).toFixed(1)}s`);
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(xhr.responseText);
          } else {
            let detail = `Erreur ${xhr.status}`;
            try {
              const j = JSON.parse(xhr.responseText);
              if (j.detail) detail = j.detail;
            } catch {}
            reject(new Error(detail));
          }
        };
        xhr.send(formData);
      });

      // For premium video flow, the upload returns a job_id and we poll.
      // For standard 4-photo flow, the upload returns the full result directly.
      let data: AnalysisResult;
      if (mode === "premium") {
        const submit = JSON.parse(uploadResponseText);
        if (!submit.job_id) {
          throw new Error("Reponse submit invalide (job_id manquant)");
        }
        const jobId: string = submit.job_id;
        setPollJobId(jobId);
        setPollModalUrl(MODAL_URL);
        try {
          sessionStorage.setItem("eteka_pending_job", JSON.stringify({ jobId, modalUrl: MODAL_URL, ts: Date.now() }));
        } catch {}
        console.log(`[Body Scan] Submitted, polling job ${jobId}...`);

        data = await pollJob(MODAL_URL, jobId, setPollSec);
        try { sessionStorage.removeItem("eteka_pending_job"); } catch {}
      } else {
        try {
          data = JSON.parse(uploadResponseText);
        } catch {
          throw new Error("Reponse JSON invalide");
        }
      }

      console.log("[Body Scan] Result:", {
        hasMeasurements: !!data.measurements,
        measurementKeys: data.measurements ? Object.keys(data.measurements) : [],
        hasVertices: !!data.vertices,
        verticesCount: data.vertices?.length || 0,
        hasFaces: !!data.faces,
        facesCount: data.faces?.length || 0,
        hasSlices: !!data.slices,
        sliceKeys: data.slices ? Object.keys(data.slices) : [],
      });

      if (!data.measurements) {
        throw new Error("Reponse invalide: mesures manquantes");
      }

      setResult(data);
      setStatus("done");
    } catch (err) {
      console.error("[Body Scan] Analyze failed:", err);
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
    <div className={`flex flex-col ${embedMode ? "" : "min-h-screen"}`}>
      {!embedMode && (
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
      )}

      {embedMode && status !== "idle" && (
        <div className="flex justify-end px-4 pt-3">
          <button onClick={handleReset} className="text-xs text-[var(--primary-light)] hover:underline">
            Recommencer
          </button>
        </div>
      )}

      <main ref={mainRef} className="flex-1 px-4 py-6 max-w-5xl mx-auto w-full">
        {/* Mode selection */}
        {status === "idle" && (
          <div className="space-y-6 max-w-2xl mx-auto">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold">Analysez votre morphologie</h2>
              <p className="text-sm text-[var(--foreground)]/60">
                Filmez-vous en 360&deg; pendant ~15 secondes pour une reconstruction 3D
              </p>
            </div>
            <button
              onClick={() => {
                setMode("premium");
                setStatus("instructions");
              }}
              className="glass rounded-2xl p-8 w-full text-center hover:bg-[var(--surface-light)] transition-colors border-2 border-[var(--accent)]/30"
            >
              <h3 className="font-semibold text-[var(--accent)] text-xl mb-2">Commencer le scan</h3>
              <p className="text-sm text-[var(--foreground)]/70">
                Une vidéo 360&deg; de 15 secondes autour du corps
              </p>
            </button>
          </div>
        )}

        {/* Instructions */}
        {status === "instructions" && (
          <CaptureInstructions
            onContinue={() => setStatus("capturing")}
            onBack={() => setStatus("idle")}
          />
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

        {/* Error */}
        {status === "error" && (
          <div className="max-w-md mx-auto space-y-4 text-center">
            <div className="bg-[var(--error)]/10 border border-[var(--error)]/30 rounded-xl p-4 text-sm text-[var(--error)]">
              {error || "Une erreur est survenue"}
            </div>
            <button
              onClick={() => {
                setError("");
                setStatus("form");
              }}
              className="w-full py-3 rounded-xl font-semibold text-white bg-[var(--primary)] hover:bg-[var(--primary-light)] transition-all"
            >
              Reessayer
            </button>
            <button
              onClick={handleReset}
              className="text-sm text-[var(--foreground)]/60 hover:text-[var(--foreground)] underline"
            >
              Recommencer a zero
            </button>
          </div>
        )}

        {/* Analyzing */}
        {status === "analyzing" && (
          <div className="max-w-md mx-auto space-y-4 text-center">
            {previewUrl && (
              <div className="relative w-48 mx-auto rounded-xl overflow-hidden">
                {mode === "premium" ? (
                  <video
                    src={previewUrl}
                    autoPlay
                    loop
                    muted
                    playsInline
                    className="w-full object-contain rounded-xl opacity-70"
                  />
                ) : (
                  <img src={previewUrl} alt="Analyse" className="w-full object-contain rounded-xl opacity-70" />
                )}
                <div className="scan-line absolute left-0 w-full h-0.5 bg-[var(--accent)] shadow-[0_0_12px_var(--accent),0_0_24px_var(--accent)]" />
              </div>
            )}
            {analyzePhase === "extract" ? (
              <>
                <p className="text-sm text-[var(--foreground)]/60">
                  Extraction des images depuis la vidéo...
                </p>
                <div className="w-full bg-[var(--surface)] rounded-full h-2 overflow-hidden">
                  <div
                    className="h-2 bg-[var(--accent)] transition-all"
                    style={{ width: `${extractPct}%` }}
                  />
                </div>
                <p className="text-xs text-[var(--foreground)]/40">{extractPct}%</p>
              </>
            ) : analyzePhase === "upload" ? (
              <>
                <p className="text-sm text-[var(--foreground)]/60">
                  Envoi des images au serveur...
                </p>
                <div className="w-full bg-[var(--surface)] rounded-full h-2 overflow-hidden">
                  <div
                    className="h-2 bg-[var(--accent)] transition-all"
                    style={{ width: `${uploadPct}%` }}
                  />
                </div>
                <p className="text-xs text-[var(--foreground)]/40">{uploadPct}%</p>
              </>
            ) : (
              <>
                <p className="text-sm text-[var(--foreground)]/60">
                  Reconstruction 3D multi-vue en cours...
                </p>
                {mode === "premium" && (
                  <div className="bg-[var(--surface)] rounded-xl p-3 mt-2 text-left text-xs space-y-1 font-mono">
                    <div className="flex justify-between">
                      <span className="text-[var(--foreground)]/60">Temps</span>
                      <span className="text-[var(--accent)]">
                        {Math.floor(pollSec / 60)}min {pollSec % 60}s
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--foreground)]/60">Polls</span>
                      <span>{pollCount}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--foreground)]/60">Statut</span>
                      <span className="truncate max-w-[60%]">{pollStatus}</span>
                    </div>
                    {pollJobId && (
                      <div className="text-[var(--foreground)]/40 truncate">{pollJobId}</div>
                    )}
                  </div>
                )}
                {mode === "premium" && pollJobId && (
                  <button
                    onClick={manualCheck}
                    className="w-full mt-2 py-2 rounded-lg bg-[var(--accent)] text-black text-sm font-medium hover:opacity-90 active:scale-[0.98] transition-all"
                  >
                    Vérifier maintenant
                  </button>
                )}
                <p className="text-xs text-[var(--foreground)]/40">
                  {mode === "premium" ? "1-3 min. Garde l'onglet ouvert au premier plan." : "Cela peut prendre 30-60 secondes"}
                </p>
              </>
            )}
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
                uvs={result.uvs}
                textureB64={result.texture_b64}
              />
            )}

            {/* 4 photos grid (Standard mode) for comparison with 3D model */}
            {mode === "standard" && photos && (
              <div className="glass rounded-2xl p-4">
                <div className="text-sm font-semibold mb-3 text-[var(--foreground)]/70 text-center">
                  Vos 4 photos
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {(["front", "left", "back", "right"] as const).map((key) => {
                    const labels: Record<string, string> = {
                      front: "Face",
                      left: "Profil G",
                      back: "Dos",
                      right: "Profil D",
                    };
                    return (
                      <div key={key} className="space-y-1">
                        <img
                          src={photos[key].preview}
                          alt={labels[key]}
                          className="w-full aspect-[3/4] object-cover rounded-lg"
                        />
                        <div className="text-xs text-center text-[var(--foreground)]/50">
                          {labels[key]}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-3 text-center text-sm text-[var(--foreground)]/60">
                  {height} cm / {weight} kg / {gender}
                </div>
              </div>
            )}

            {mode === "premium" && previewUrl && (
              <div className="glass rounded-2xl p-4">
                <video src={previewUrl} controls className="w-full max-h-[400px] rounded-xl" />
                <div className="mt-3 text-center text-sm text-[var(--foreground)]/60">
                  {height} cm / {weight} kg / {gender}
                </div>
              </div>
            )}

            <MeasurementResults measurements={result.measurements} />
          </div>
        )}
      </main>

      {!embedMode && (
        <footer className="px-4 py-3 text-center text-xs text-[var(--foreground)]/30">
          ETEKA Body Scan
        </footer>
      )}
    </div>
  );
}
