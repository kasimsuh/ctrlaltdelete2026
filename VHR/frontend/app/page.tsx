"use client";

import { useEffect, useRef, useState } from "react";

type RppgResponse = {
  avg_hr_bpm: number | null;
  hr_quality: "low" | "medium" | "high";
  usable_seconds: number;
  bpm_series?: number[];
  engine?: string;
  note?: string;
  upload_mb?: number;
  timing_ms?: {
    upload_write?: number;
    preprocess?: number;
    analysis?: number;
    total?: number;
  };
};

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

const MIME_TYPES = [
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
  "video/mp4"
];

function getSupportedMimeType(): string {
  if (typeof window === "undefined" || typeof MediaRecorder === "undefined") {
    return "";
  }

  return MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [autoUpload, setAutoUpload] = useState(true);
  const [statusMessage, setStatusMessage] = useState("Idle");
  const [lastResult, setLastResult] = useState<RppgResponse | null>(null);

  useEffect(() => {
    if (!videoRef.current || !stream) {
      return;
    }

    videoRef.current.srcObject = stream;
    videoRef.current.play().catch(() => {
      // Ignore autoplay blocks; user action will start playback.
    });
  }, [stream]);

  useEffect(() => {
    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [stream]);

  const startCamera = async () => {
    try {
      const newStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640, max: 960 },
          height: { ideal: 480, max: 720 },
          frameRate: { ideal: 30, max: 30 }
        },
        audio: false
      });

      setStream(newStream);
      setStatusMessage("Camera ready.");
    } catch (error) {
      console.error("Failed to start camera", error);
      setStatusMessage("Camera permission failed.");
    }
  };

  const startRecording = () => {
    if (!stream) {
      setStatusMessage("Start camera first.");
      return;
    }

    const mimeType = getSupportedMimeType();
    const recorderOptions: MediaRecorderOptions = {
      videoBitsPerSecond: 1_200_000
    };
    if (mimeType) {
      recorderOptions.mimeType = mimeType;
    }
    const recorder = new MediaRecorder(stream, recorderOptions);

    chunksRef.current = [];

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, {
        type: recorder.mimeType || "video/webm"
      });

      setRecordedBlob(blob);
      setStatusMessage(
        `Recording complete (${(blob.size / 1024 / 1024).toFixed(2)} MB).`
      );

      if (autoUpload) {
        await uploadClip(blob);
      }
    };

    recorder.start(1000);
    mediaRecorderRef.current = recorder;
    setIsRecording(true);
    setStatusMessage("Recording in progress...");
  };

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state !== "recording") {
      setStatusMessage("Nothing is recording.");
      return;
    }

    recorder.stop();
    setIsRecording(false);
  };

  const uploadClip = async (blobOverride?: Blob) => {
    const clip = blobOverride ?? recordedBlob;

    if (!clip) {
      setStatusMessage("No clip available yet.");
      return;
    }

    const extension = clip.type.includes("mp4") ? "mp4" : "webm";
    const fileName = `session-${Date.now()}.${extension}`;
    const clipSizeMb = clip.size / 1024 / 1024;
    const formData = new FormData();
    formData.append("video", clip, fileName);

    setIsUploading(true);
    setStatusMessage(`Uploading ${clipSizeMb.toFixed(2)} MB clip to backend...`);

    try {
      const requestStarted = performance.now();
      const response = await fetch(`${BACKEND_URL}/rppg`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Upload failed (${response.status})`);
      }

      const json = (await response.json()) as RppgResponse;
      const roundTripMs = Math.round(performance.now() - requestStarted);
      console.log("rPPG result", {
        ...json,
        client_roundtrip_ms: roundTripMs,
        client_clip_mb: Number(clipSizeMb.toFixed(2))
      });
      setLastResult(json);
      const backendTotalMs = json.timing_ms?.total;
      const backendText = backendTotalMs
        ? ` Backend total: ${(backendTotalMs / 1000).toFixed(2)}s.`
        : "";
      setStatusMessage(
        `Upload complete in ${(roundTripMs / 1000).toFixed(2)}s.${backendText} Check browser console for full payload.`
      );
    } catch (error) {
      console.error("Upload error", error);
      setStatusMessage("Upload failed; check backend logs.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <main className="page">
      <section className="card">
        <p className="eyebrow">VHR Spike Demo</p>
        <h1>Live Video + Upload + open-rppg</h1>
        <p className="subtitle">
          Run a 20-30s camera capture while the Gemini placeholder is active.
        </p>

        <div className="grid">
          <div className="videoWrap">
            <video ref={videoRef} muted playsInline autoPlay />
            <div className="geminiBox">
              <span className="pulse" /> Gemini placeholder: "I am speaking..."
            </div>
          </div>

          <div className="controls">
            <button onClick={startCamera} disabled={!!stream || isUploading}>
              Start Camera
            </button>
            <button
              onClick={startRecording}
              disabled={!stream || isRecording || isUploading}
            >
              Start Recording
            </button>
            <button onClick={stopRecording} disabled={!isRecording || isUploading}>
              Stop Recording
            </button>
            <button
              onClick={() => void uploadClip()}
              disabled={!recordedBlob || isUploading || isRecording}
            >
              Upload Clip
            </button>

            <label className="toggle">
              <input
                type="checkbox"
                checked={autoUpload}
                onChange={(event) => setAutoUpload(event.target.checked)}
              />
              Auto-upload when recording stops
            </label>

            <p className="status">{statusMessage}</p>

            {lastResult ? (
              <pre>
                {JSON.stringify(
                  {
                    avg_hr_bpm: lastResult.avg_hr_bpm,
                    hr_quality: lastResult.hr_quality,
                    usable_seconds: lastResult.usable_seconds,
                    engine: lastResult.engine,
                    upload_mb: lastResult.upload_mb,
                    timing_ms: lastResult.timing_ms
                  },
                  null,
                  2
                )}
              </pre>
            ) : null}
          </div>
        </div>
      </section>
    </main>
  );
}
