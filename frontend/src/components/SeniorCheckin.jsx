import { useEffect, useMemo, useRef, useState } from "react";

import useCheckin from "../hooks/useCheckin.js";
import { cameraDurationMs, statusColor } from "../lib/screening.js";
import CompletionScreen from "./checkin/CompletionScreen.jsx";

const PAGE_SHELL_CLASS =
  "min-h-screen bg-[#f3f0ea] px-3 py-2 text-[#1d1b19] sm:px-4 sm:py-3";

const FACE_CAPTURE_MS = cameraDurationMs;
const CAMERA_DONE_STATUSES = new Set(["Uploaded", "Error"]);

function HeaderBar({ authUser, logout }) {
  return (
    <header className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-[22px] border border-[#e8e2d8] bg-[#f7f7f7] px-5 py-4 shadow-[0_14px_28px_rgba(44,39,34,0.08)] sm:mt-4 sm:px-6">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[#9b6a52]">
          Guardian Check-In
        </p>
        <p className="mt-1 text-sm text-stone-700">
          Signed in as{" "}
          <span className="font-semibold">
            {authUser?.firstName} {authUser?.lastName}
          </span>
        </p>
      </div>
      <button
        type="button"
        onClick={logout}
        className="rounded-full bg-stone-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-stone-800"
      >
        Log out
      </button>
    </header>
  );
}

function Mascot() {
  return (
    <div className="grid h-full w-full place-items-center">
      <div className="relative h-44 w-44">
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-[#f2d4bf] via-[#f4e2d5] to-[#f7f7f7]" />
        <div className="absolute left-6 top-7 h-20 w-20 rounded-full bg-[#f7f7f7] shadow-[0_10px_18px_rgba(60,45,36,0.08)]" />
        <div className="absolute right-8 top-10 h-16 w-16 rounded-full bg-[#f7f7f7] shadow-[0_10px_18px_rgba(60,45,36,0.08)]" />
        <div className="absolute bottom-7 left-1/2 h-14 w-28 -translate-x-1/2 rounded-[999px] bg-[#e46535]/15 shadow-inner" />
        <div className="absolute bottom-11 left-1/2 h-2 w-10 -translate-x-1/2 rounded-full bg-[#e46535]" />
      </div>
    </div>
  );
}

function ProgressRing({ percent }) {
  const radius = 30;
  const stroke = 6;
  const normalizedRadius = radius - stroke * 0.5;
  const circumference = normalizedRadius * 2 * Math.PI;
  const offset = circumference - (Math.min(100, Math.max(0, percent)) / 100) * circumference;

  return (
    <svg
      width={(radius + stroke) * 2}
      height={(radius + stroke) * 2}
      className="text-[#e46535]"
    >
      <circle
        stroke="rgba(148, 163, 184, 0.35)"
        fill="transparent"
        strokeWidth={stroke}
        r={normalizedRadius}
        cx={radius + stroke}
        cy={radius + stroke}
      />
      <circle
        stroke="currentColor"
        fill="transparent"
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={`${circumference} ${circumference}`}
        strokeDashoffset={offset}
        r={normalizedRadius}
        cx={radius + stroke}
        cy={radius + stroke}
      />
    </svg>
  );
}

function HeroStage({
  showMascot,
  cameraVideoRef,
  isRunning,
  progressPercent,
  timerDone,
  isFaceScanActive,
}) {
  return (
    <div className="w-full overflow-hidden rounded-[28px] border border-[#e8e2d8] bg-[#1f1a17] shadow-[0_18px_36px_rgba(44,39,34,0.18)]">
      <div className="relative aspect-video w-full">
        {showMascot || !isRunning ? (
          <Mascot />
        ) : (
          <video
            ref={cameraVideoRef}
            autoPlay
            muted
            playsInline
            className="h-full w-full object-cover"
          />
        )}

        {isRunning ? (
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-3 bg-gradient-to-t from-black/60 via-black/25 to-transparent px-5 py-4 text-[#fff6e8]">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[#f8e7d3]/80">
                Face scan
              </p>
              <p className="mt-1 text-sm font-medium">
                {timerDone
                  ? "Finishing analysis..."
                  : isFaceScanActive
                    ? "Analyzing face data for 10 seconds..."
                    : "Waiting for camera..."}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <ProgressRing percent={progressPercent} />
              <p className="text-sm font-semibold tabular-nums">
                {progressPercent}%
              </p>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ActionRow({
  onStart,
  onSecondary,
  startDisabled,
  secondaryDisabled,
  startLabel,
  secondaryLabel,
}) {
  return (
    <div className="flex w-full max-w-md flex-wrap justify-center gap-3">
      <button
        type="button"
        onClick={onStart}
        disabled={startDisabled}
        className="rounded-full bg-gradient-to-b from-[#e46535] to-[#d8542a] px-6 py-3 text-base font-semibold text-white shadow-[0_10px_18px_rgba(222,91,47,0.28)] transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-70"
      >
        {startLabel}
      </button>
      <button
        type="button"
        onClick={onSecondary}
        disabled={secondaryDisabled}
        className="rounded-full border border-[#cfc4b6] bg-white px-6 py-3 text-base font-semibold text-stone-700 transition hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {secondaryLabel}
      </button>
    </div>
  );
}

export default function SeniorCheckin({ authUser, authToken, logout }) {
  const {
    status,
    reason,
    isVoiceLive,
    isCheckinComplete,
    voiceStatus,
    voiceLog,
    cameraStatus,
    facialSymmetryStatus,
    facialSymmetryReason,
    cameraVideoRef,
    startVoice,
    stopVoice,
  } = useCheckin(authUser, authToken);

  const [phase, setPhase] = useState("idle"); // idle | running | mascot | complete
  const [elapsedMs, setElapsedMs] = useState(0);
  const [timerDone, setTimerDone] = useState(false);
  const [isStartLocked, setIsStartLocked] = useState(false);
  const [isPermissionChecking, setIsPermissionChecking] = useState(false);
  const [permissionError, setPermissionError] = useState("");

  const runStartAtRef = useRef(null);
  const intervalRef = useRef(null);
  const timeoutRef = useRef(null);
  const startLockRef = useRef(false);

  const clearRunTimers = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      clearRunTimers();
    };
  }, []);

  useEffect(() => {
    if (phase !== "running") return;
    if (cameraStatus !== "Recording 10s..." || runStartAtRef.current) return;

    runStartAtRef.current = Date.now();
    clearRunTimers();
    intervalRef.current = window.setInterval(() => {
      if (!runStartAtRef.current) return;
      const elapsed = Math.min(Date.now() - runStartAtRef.current, FACE_CAPTURE_MS);
      setElapsedMs(elapsed);
    }, 100);
    timeoutRef.current = window.setTimeout(() => {
      setElapsedMs(FACE_CAPTURE_MS);
      setTimerDone(true);
      clearRunTimers();
    }, FACE_CAPTURE_MS);
  }, [cameraStatus, phase]);

  useEffect(() => {
    if (phase !== "running") return;
    if (cameraStatus !== "Error") return;
    setElapsedMs(FACE_CAPTURE_MS);
    setTimerDone(true);
  }, [cameraStatus, phase]);

  useEffect(() => {
    if (!isCheckinComplete || isVoiceLive) return;
    clearRunTimers();
    runStartAtRef.current = null;
    setElapsedMs(FACE_CAPTURE_MS);
    setTimerDone(true);
    setPhase("complete");
    setIsStartLocked(false);
    setIsPermissionChecking(false);
    setPermissionError("");
    startLockRef.current = false;
  }, [isCheckinComplete, isVoiceLive]);

  useEffect(() => {
    if (timerDone && CAMERA_DONE_STATUSES.has(cameraStatus)) {
      setPhase("mascot");
      setIsStartLocked(false);
      startLockRef.current = false;
      clearRunTimers();
    }
  }, [cameraStatus, timerDone]);

  const handleStart = async () => {
    if (startLockRef.current || isStartLocked || phase === "running") return;
    setPermissionError("");
    setIsPermissionChecking(true);

    let preflightStream;
    try {
      if (!navigator?.mediaDevices?.getUserMedia) {
        throw new Error("Media devices API is not available in this browser.");
      }
      preflightStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
    } catch {
      setPermissionError("Camera and microphone access is required to start.");
      setIsPermissionChecking(false);
      return;
    } finally {
      if (preflightStream) {
        preflightStream.getTracks().forEach((track) => track.stop());
      }
    }

    setIsPermissionChecking(false);
    startLockRef.current = true;
    setIsStartLocked(true);
    setPhase("running");
    setElapsedMs(0);
    setTimerDone(false);
    runStartAtRef.current = null;
    clearRunTimers();
    void startVoice();
  };

  const handleSecondary = async () => {
    if (isVoiceLive) {
      await stopVoice();
    }
    clearRunTimers();
    runStartAtRef.current = null;
    setElapsedMs(0);
    setTimerDone(false);
    setPhase("idle");
    setIsStartLocked(false);
    setIsPermissionChecking(false);
    setPermissionError("");
    startLockRef.current = false;
  };

  const progressPercent = useMemo(
    () => Math.round((Math.min(elapsedMs, FACE_CAPTURE_MS) / FACE_CAPTURE_MS) * 100),
    [elapsedMs],
  );
  const isRunning = phase === "running";
  const showMascot = phase === "mascot" || phase === "idle";
  const isFaceScanActive = isRunning && cameraStatus === "Recording 10s...";

  if (phase === "complete") {
    return (
      <div>
        <CompletionScreen />
        <div className="mx-auto -mt-10 w-full max-w-3xl px-3 pb-10 sm:px-4">
          <div className="rounded-[22px] border border-[#e8e2d8] bg-white p-5 shadow-[0_16px_30px_rgba(44,39,34,0.1)]">
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-stone-500">
              Result
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <span
                className={`rounded-full px-4 py-2 text-sm font-semibold ${
                  statusColor[status] || statusColor.neutral
                }`}
              >
                {status || "—"}
              </span>
              <span className="text-sm text-stone-600">{reason}</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={PAGE_SHELL_CLASS}>
      <main className="mx-auto flex min-h-[calc(100svh-1rem)] w-full max-w-5xl flex-col sm:min-h-[calc(100svh-1.5rem)]">
        <HeaderBar authUser={authUser} logout={logout} />

        <section className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center gap-5 py-6 sm:gap-6">
          <HeroStage
            showMascot={showMascot}
            cameraVideoRef={cameraVideoRef}
            isRunning={isRunning}
            progressPercent={progressPercent}
            timerDone={timerDone}
            isFaceScanActive={isFaceScanActive}
          />

          <div className="w-full rounded-[22px] border border-[#e8e2d8] bg-[#f7f7f7] p-5 shadow-[0_14px_28px_rgba(44,39,34,0.08)]">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-stone-500">
                  Session
                </p>
                <p className="mt-1 text-sm font-medium text-stone-800">
                  {isVoiceLive ? `Voice: ${voiceStatus}` : "Ready"}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-stone-500">
                  Camera
                </p>
                <p className="mt-1 text-sm font-medium text-stone-800">
                  {cameraStatus}
                </p>
              </div>
            </div>
            <p className="mt-3 text-sm text-stone-600">
              Facial symmetry:{" "}
              <span className="font-medium text-stone-800">
                {facialSymmetryStatus}
              </span>
              {facialSymmetryReason ? (
                <span className="text-stone-500"> · {facialSymmetryReason}</span>
              ) : null}
            </p>
          </div>

          <ActionRow
            onStart={handleStart}
            onSecondary={handleSecondary}
            startDisabled={isVoiceLive || isStartLocked || isPermissionChecking}
            secondaryDisabled={!isVoiceLive && phase === "idle"}
            startLabel={
              isPermissionChecking
                ? "Checking Access..."
                : isRunning
                  ? "Running..."
                  : "Start Check-In"
            }
            secondaryLabel={isVoiceLive || isRunning ? "Stop Session" : "Reset"}
          />

          {permissionError ? (
            <p className="text-center text-sm font-medium text-rose-700">
              {permissionError}
            </p>
          ) : null}

          <details className="w-full max-w-3xl rounded-[22px] border border-[#e8e2d8] bg-white px-5 py-4 text-sm text-stone-700 shadow-[0_14px_28px_rgba(44,39,34,0.06)]">
            <summary className="cursor-pointer select-none text-xs font-semibold uppercase tracking-[0.24em] text-stone-500">
              Debug log
            </summary>
            <div className="mt-3 space-y-2">
              {voiceLog.length === 0 ? (
                <p className="text-stone-500">No messages yet.</p>
              ) : (
                voiceLog.slice(-30).map((line, idx) => (
                  <p key={`${idx}-${line}`} className="break-words">
                    {line}
                  </p>
                ))
              )}
            </div>
          </details>
        </section>
      </main>
    </div>
  );
}

