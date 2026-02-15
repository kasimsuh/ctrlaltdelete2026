import { useEffect, useRef, useState } from "react";

import useCheckin from "../hooks/useCheckin.js";
import CompletionScreen from "./checkin/CompletionScreen.jsx";

const CAMERA_DONE_STATUSES = new Set(["Recorded", "Uploading...", "Error"]);
const FACE_CAPTURE_MS = 10000;
const PAGE_SHELL_CLASS =
  "min-h-screen bg-[#f3f0ea] px-3 py-2 text-[#1d1b19] sm:px-4 sm:py-3";

function HeaderBar({ authUser, logout }) {
  const displayName =
    [authUser?.firstName, authUser?.lastName]
      .filter(Boolean)
      .join(" ")
      .trim() || "there";

  return (
    <div className="flex items-center justify-between rounded-full border border-[#e8e2d8] bg-[#f7f7f7] px-4 py-2 text-sm text-stone-700">
      <div className="flex items-center gap-2.5">
        <img
          src="/senicarelogo.png"
          alt="SeniCare logo"
          className="h-8 w-8 rounded-full object-cover"
        />
        <span className="text-base font-medium">
          <span className="font-semibold text-stone-900">Welcome,</span>{" "}
          {displayName}!
        </span>
      </div>
      <button
        type="button"
        onClick={logout}
        className="rounded-full bg-[#171513] px-4 py-2 text-sm font-semibold text-white transition hover:bg-stone-800"
      >
        Log out
      </button>
    </div>
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
  const showVideoPlaceholder = !showMascot && !isFaceScanActive;

  return (
    <div className="w-full rounded-[22px] border border-[#e8e2d8] bg-[#f7f7f7] p-3 shadow-[0_12px_24px_rgba(44,39,34,0.08)] sm:p-4">
      <div className="relative aspect-video overflow-hidden rounded-2xl border border-[#ddd3c6] bg-[#ece7de]">
        {showMascot ? (
          <div className="grid h-full w-full place-items-center bg-[radial-gradient(circle_at_top,_#f8efe1_0%,_#efe4d3_100%)] p-6">
            <img
              src="/senicarelogo.png"
              alt="SeniCare mascot"
              className="h-full max-h-64 w-auto rounded-2xl object-contain shadow-[0_10px_20px_rgba(44,39,34,0.12)]"
            />
          </div>
        ) : (
          <video
            ref={cameraVideoRef}
            autoPlay
            muted
            playsInline
            className="h-full w-full object-cover"
          />
        )}
        {showVideoPlaceholder ? (
          <div className="absolute inset-0 grid place-items-center bg-[radial-gradient(circle_at_top,_#f8efe1_0%,_#efe4d3_100%)] p-6">
            <img
              src="/senicarelogo.png"
              alt="SeniCare logo"
              className="h-full max-h-64 w-auto rounded-2xl object-contain shadow-[0_10px_20px_rgba(44,39,34,0.12)]"
            />
          </div>
        ) : null}
        {isRunning ? (
          <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[rgba(26,23,20,0.65)] to-transparent p-4">
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-[#f0e6d8]">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[#e46535] to-[#d8542a] transition-[width] duration-100 ease-linear"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <p className="mt-2 text-xs font-medium text-[#fff6e8]">
              {timerDone
                ? "Finishing analysis..."
                : isFaceScanActive
                  ? "Analyzing face data for 10 seconds..."
                  : "Loading Questions ..."}
            </p>
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
    isVoiceLive,
    isCheckinComplete,
    cameraStatus,
    cameraVideoRef,
    startVoice,
    stopVoice,
  } = useCheckin(authUser, authToken);
  const [phase, setPhase] = useState("idle");
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
      const elapsed = Math.min(
        Date.now() - runStartAtRef.current,
        FACE_CAPTURE_MS,
      );
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
    } catch (error) {
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

  const progressPercent = Math.round(
    (Math.min(elapsedMs, FACE_CAPTURE_MS) / FACE_CAPTURE_MS) * 100,
  );
  const isRunning = phase === "running";
  const showMascot = phase === "mascot";
  const isComplete = phase === "complete";
  const isFaceScanActive = isRunning && cameraStatus === "Recording 10s...";

  if (isComplete) {
    return <CompletionScreen />;
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

          <ActionRow
            onStart={handleStart}
            onSecondary={handleSecondary}
            startDisabled={isVoiceLive || isStartLocked || isPermissionChecking}
            secondaryDisabled={!isVoiceLive && !showMascot && !isRunning}
            startLabel={
              isPermissionChecking
                ? "Checking Access..."
                : showMascot
                  ? "Start Again"
                  : isRunning
                    ? "Running..."
                    : "Start Check-In"
            }
            secondaryLabel={
              isVoiceLive || isRunning ? "Stop Session" : "Show Camera"
            }
          />

          {permissionError ? (
            <p className="text-center text-sm font-medium text-rose-700">
              {permissionError}
            </p>
          ) : null}
        </section>
      </main>
    </div>
  );
}
