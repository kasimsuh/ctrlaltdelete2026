import { useRef, useState } from "react";
import { Conversation } from "@elevenlabs/client";

import { API_BASE } from "../lib/api.js";
import {
  completionPhrase,
  cameraDurationMs,
  normalizeAnswer,
  INITIAL_RESPONSES,
  parseSymptomTranscript,
  buildTranscript,
} from "../lib/screening.js";

const apiBase = API_BASE;

export default function useCheckin(authUser, authToken) {
  const [status, setStatus] = useState(null);
  const [reason, setReason] = useState("Run a check-in to see triage output.");
  const [isDemoMode, setIsDemoMode] = useState(true);
  const [voiceStatus, setVoiceStatus] = useState("Idle");
  const [voiceLog, setVoiceLog] = useState([]);
  const [isVoiceLive, setIsVoiceLive] = useState(false);

  const [cameraStatus, setCameraStatus] = useState("Idle");
  const [facialSymmetryStatus, setFacialSymmetryStatus] = useState("Not run");
  const [facialSymmetryReason, setFacialSymmetryReason] = useState(
    "Facial symmetry results will appear after camera upload.",
  );
  const cameraVideoRef = useRef(null);
  const cameraStreamRef = useRef(null);

  const convoRef = useRef(null);
  const checkinIdRef = useRef(null);
  const isSessionOpenRef = useRef(false);
  const startVoiceInFlightRef = useRef(false);
  const lastUserMessageRef = useRef(null);
  const lastModeRef = useRef("listening");
  const finalizePendingRef = useRef(false);
  const finalizeStartedRef = useRef(false);
  const finalizeTimeoutRef = useRef(null);
  const cameraStartedRef = useRef(false);
  const introPendingRef = useRef(false);
  const currentQuestionIndexRef = useRef(0);
  const responsesRef = useRef(INITIAL_RESPONSES.map((item) => ({ ...item })));

  // Note: we no longer create a "dummy" check-in here. That was generating an extra
  // check-in history entry. The real flow happens via `startVoice()`.

  const stopVoice = async () => {
    // If the ElevenLabs SDK reports disconnect reason "user", it means we called endSession().
    // Stack logging helps identify the caller (button click vs auto-stop).
    console.warn("[ElevenLabs] stopVoice() called", new Error().stack);
    setVoiceStatus("Stopping...");
    isSessionOpenRef.current = false;
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    if (cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = null;
    }
    try {
      await convoRef.current?.endSession?.();
    } catch {
      // ignore
    }
    convoRef.current = null;
    setIsVoiceLive(false);
    setVoiceStatus("Idle");
    startVoiceInFlightRef.current = false;
  };

  const finalizeSession = async ({ uploadPromise }) => {
    if (finalizeStartedRef.current) return;
    finalizeStartedRef.current = true;
    if (finalizeTimeoutRef.current) {
      clearTimeout(finalizeTimeoutRef.current);
      finalizeTimeoutRef.current = null;
    }

    setVoiceLog((prev) => [...prev, "[System] Finalizing check-in..."]);
    try {
      await uploadPromise;
    } catch {
      // ignore
    }
    try {
      await saveQaJson();
    } catch {
      // ignore
    }
    try {
      await completeCheckin();
    } catch (e) {
      setVoiceLog((prev) => [
        ...prev,
        `[System] Complete check-in failed: ${e?.message || "unknown error"}`,
      ]);
    }

    // Give the agent audio path a beat to flush before closing.
    setTimeout(() => {
      stopVoice();
    }, 1200);
  };

  const captureAndUploadCamera = async ({ checkinId, uploadUrl, uploadHeaders }) => {
    try {
      setCameraStatus("Recording 10s...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      cameraStreamRef.current = stream;
      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = stream;
      }

      const preferredType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
        ? "video/webm;codecs=vp8"
        : "video/webm";
      const recorder = new MediaRecorder(
        stream,
        preferredType ? { mimeType: preferredType } : undefined,
      );
      const chunks = [];

      const videoBlob = await new Promise((resolve, reject) => {
        recorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) chunks.push(event.data);
        };
        recorder.onerror = (event) =>
          reject(event.error || new Error("Camera recording failed"));
        recorder.onstop = () =>
          resolve(new Blob(chunks, { type: recorder.mimeType || "video/webm" }));
        recorder.start();
        setTimeout(() => recorder.stop(), cameraDurationMs);
      });

      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach((track) => track.stop());
        cameraStreamRef.current = null;
      }
      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = null;
      }

      setCameraStatus("Uploading...");
      const formData = new FormData();
      formData.append("video", videoBlob, "checkin.webm");
      formData.append("metadata", JSON.stringify({ duration_ms: cameraDurationMs }));

      const resp = await fetch(uploadUrl, {
        method: "POST",
        headers: uploadHeaders,
        body: formData,
      });
      if (!resp.ok) throw new Error("Failed to upload camera clip");

      const uploadData = await resp.json();
      const facial = uploadData?.facial_symmetry;
      if (facial?.status) {
        setFacialSymmetryStatus(facial.status);
        setFacialSymmetryReason(facial.reason || "No details returned.");
      } else {
        setFacialSymmetryStatus("Missing");
        setFacialSymmetryReason("No facial symmetry payload returned.");
      }
      setCameraStatus("Uploaded");
    } catch (e) {
      setCameraStatus("Error");
      setFacialSymmetryStatus("Error");
      setFacialSymmetryReason(e?.message || "Camera capture/upload failed.");
      throw e;
    }
  };

  const saveQaJson = async () => {
    // Ensure the saved JSON always includes the canonical question text.
    const items = responsesRef.current.map((r, idx) => ({
      question: r?.q ?? INITIAL_RESPONSES[idx]?.q ?? "",
      answer: r?.transcript ?? "",
    }));

    await fetch(`${apiBase}/stt/qa/save`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
      },
      body: JSON.stringify({
        items,
      }),
    });
  };

  const completeCheckin = async () => {
    const checkinId = checkinIdRef.current;
    if (!checkinId) return;

    const responses = responsesRef.current;
    const symptomTranscript = responses[1]?.transcript || "";
    const parsedSymptoms = parseSymptomTranscript(symptomTranscript);
    const symptomAnswer = responses[1]?.answer;

    const answers = {
      dizziness:
        parsedSymptoms.dizziness ||
        (symptomAnswer === true &&
          !parsedSymptoms.chest_pain &&
          !parsedSymptoms.trouble_breathing),
      chest_pain: parsedSymptoms.chest_pain,
      trouble_breathing: parsedSymptoms.trouble_breathing,
      medication_taken: responses[2]?.answer ?? null,
    };

    const transcript = buildTranscript(responses);

    const resp = await fetch(`${apiBase}/checkins/${checkinId}/complete?force=true`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
      },
      body: JSON.stringify({ answers, transcript }),
    });
    if (!resp.ok) {
      throw new Error("Failed to complete check-in");
    }
    const result = await resp.json();
    setStatus(result.triage_status);
    setReason(result.triage_reasons.join("; "));
  };

  const updateQuestionIndexFromAi = (text) => {
    const lower = (text || "").toLowerCase();
    if (!lower) return;
    if (lower.includes("how are you feeling")) {
      currentQuestionIndexRef.current = 0;
      return;
    }
    if (
      lower.includes("dizziness") ||
      lower.includes("chest pain") ||
      lower.includes("trouble breathing") ||
      lower.includes("breathing")
    ) {
      currentQuestionIndexRef.current = 1;
      return;
    }
    if (
      lower.includes("morning medications") ||
      lower.includes("morning medication") ||
      lower.includes("take your morning")
    ) {
      currentQuestionIndexRef.current = 2;
    }
  };

  const startVoice = async () => {
    if (isVoiceLive || startVoiceInFlightRef.current) return;
    startVoiceInFlightRef.current = true;
    setIsVoiceLive(true);
    setVoiceStatus("Connecting...");
    setVoiceLog([]);
    setCameraStatus("Idle");
    setFacialSymmetryStatus("Pending");
    setFacialSymmetryReason("Camera capture in progress...");

    // Reset answers for this run.
    lastUserMessageRef.current = null;
    currentQuestionIndexRef.current = 0;
    responsesRef.current = INITIAL_RESPONSES.map((item) => ({ ...item }));
    lastModeRef.current = "listening";
    finalizePendingRef.current = false;
    finalizeStartedRef.current = false;
    cameraStartedRef.current = false;
    introPendingRef.current = false;
    if (finalizeTimeoutRef.current) {
      clearTimeout(finalizeTimeoutRef.current);
      finalizeTimeoutRef.current = null;
    }

    try {
      const checkinResponse = await fetch(`${apiBase}/checkins/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({}),
      });
      if (!checkinResponse.ok) throw new Error("Failed to start check-in");
      const checkinData = await checkinResponse.json();
      checkinIdRef.current = checkinData.checkin_id;

      // We'll start camera after the agent speaks the intro. Keep a stable promise for finalize().
      let uploadPromise = Promise.resolve();

      // Preflight mic permission before starting the ElevenLabs session (avoids permission prompts mid-connection).
      try {
        const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
        mic.getTracks().forEach((t) => t.stop());
      } catch {
        // ElevenLabs will surface its own mic error; keep going.
      }

      const signedUrlResp = await fetch(`${apiBase}/elevenlabs/signed-url`, {
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
      });
      if (!signedUrlResp.ok) throw new Error("Failed to fetch ElevenLabs signed URL");
      const { signed_url } = await signedUrlResp.json();
      if (!signed_url) throw new Error("ElevenLabs signed URL missing");

      const conversation = await Conversation.startSession({
        signedUrl: signed_url,
        onDebug: (props) => {
          // Useful for diagnosing disconnects (close codes, internal events, etc.)
          console.debug("[ElevenLabs] onDebug", props);
        },
        onStatusChange: ({ status: s }) => {
          setVoiceStatus(
            s === "connected"
              ? "Listening..."
              : s === "connecting"
                ? "Connecting..."
                : s === "disconnecting"
                  ? "Stopping..."
                  : s === "disconnected"
                    ? "Idle"
                    : s,
          );
        },
        onModeChange: ({ mode }) => {
          lastModeRef.current = mode;
          console.debug("[ElevenLabs] onModeChange", mode);

          // After the agent finishes the intro (speaking -> listening), record 10s camera.
          if (
            introPendingRef.current &&
            !cameraStartedRef.current &&
            mode === "listening"
          ) {
            introPendingRef.current = false;
            cameraStartedRef.current = true;

            uploadPromise = (async () => {
              try {
                await captureAndUploadCamera({
                  checkinId: checkinIdRef.current,
                  uploadUrl: `${apiBase}/checkins/${checkinIdRef.current}/upload`,
                  uploadHeaders: authToken ? { Authorization: `Bearer ${authToken}` } : {},
                });
              } catch {
                // Leave facial state as error; don't break the voice session.
              }

              // Continue after the 10s camera step is done.
              conversation.sendContextualUpdate(
                "Camera done. Do not ask for the camera again. Continue now with the screening questions. Ask exactly these questions in order and wait for the user's spoken answer after each: (1) How are you feeling today? (2) Are you experiencing any dizziness, chest pain, or trouble breathing? (3) Did you take your morning medications? After they answer, say exactly: \"Thank you for your responses. The screening is now complete. Goodbye.\"",
              );
            })();
          }

          if (finalizePendingRef.current && mode === "listening") {
            finalizePendingRef.current = false;
            void finalizeSession({ uploadPromise });
          }
        },
        onConnect: () => {
          isSessionOpenRef.current = true;
          setVoiceLog((prev) => [...prev, "[ElevenLabs] Connected"]);
        },
        onDisconnect: (details) => {
          isSessionOpenRef.current = false;
          console.error("[ElevenLabs] onDisconnect", details);
          setVoiceLog((prev) => [
            ...prev,
            `[ElevenLabs] Disconnected: ${details?.reason || "unknown"}`,
          ]);
          setIsVoiceLive(false);
          startVoiceInFlightRef.current = false;
        },
        onError: (message, context) => {
          setVoiceStatus("Error");
          console.error("[ElevenLabs] onError", message, context);
          setVoiceLog((prev) => [...prev, `[ElevenLabs] Error: ${message}`]);
          startVoiceInFlightRef.current = false;
        },
        onMessage: ({ source, message }) => {
          const text = String(message || "").trim();
          if (!text) return;

          if (source === "ai") {
            setVoiceLog((prev) => [...prev, `AI: ${text}`]);
            updateQuestionIndexFromAi(text);
            const answeredCount = responsesRef.current.filter(
              (item) => item.transcript,
            ).length;
            const sawCompletion =
              text.replace(/\s+/g, " ").trim() ===
              completionPhrase.replace(/\s+/g, " ").trim();
            if (sawCompletion && answeredCount >= responsesRef.current.length) {
              console.debug("[ElevenLabs] completion detected", {
                answeredCount,
                text,
              });
              // Don't end the session immediately; wait for the agent to finish speaking
              // (mode changes back to "listening"). Fallback to a timeout.
              if (lastModeRef.current === "listening") {
                void finalizeSession({ uploadPromise });
              } else {
                finalizePendingRef.current = true;
                if (finalizeTimeoutRef.current) {
                  clearTimeout(finalizeTimeoutRef.current);
                }
                finalizeTimeoutRef.current = setTimeout(() => {
                  finalizePendingRef.current = false;
                  void finalizeSession({ uploadPromise });
                }, 9000);
              }
            } else if (text.includes(completionPhrase) && !sawCompletion) {
              // Keep this visible if the SDK ever sends the phrase in a larger chunk.
              console.debug("[ElevenLabs] completion substring seen (ignored)", {
                answeredCount,
                text,
              });
            }
            return;
          }

          if (source === "user") {
            if (lastUserMessageRef.current === text) return;
            lastUserMessageRef.current = text;
            setVoiceLog((prev) => [...prev, `YOU: ${text}`]);

            const idx = currentQuestionIndexRef.current;
            const response = responsesRef.current[idx] || responsesRef.current[0];
            const mergedTranscript = response.transcript
              ? `${response.transcript} ${text}`.trim()
              : text;
            responsesRef.current[idx] = {
              ...response,
              transcript: mergedTranscript,
              answer: normalizeAnswer(idx, mergedTranscript),
            };
          }
        },
      });

      convoRef.current = conversation;

      conversation.sendContextualUpdate(
        "Say this first, then stop and wait silently: 'Hi, I am going to start your daily health check-in. Please look at the camera now and keep your face centered and still.' Do not ask any questions yet. Wait for my next instruction.",
      );
      introPendingRef.current = true;
    } catch (error) {
      setVoiceStatus("Error");
      setVoiceLog((prev) => [...prev, error?.message || "Voice setup failed."]);
      setIsVoiceLive(false);
      startVoiceInFlightRef.current = false;
    }
  };

  return {
    status,
    reason,
    isDemoMode,
    setIsDemoMode,
    voiceStatus,
    voiceLog,
    isVoiceLive,
    cameraStatus,
    facialSymmetryStatus,
    facialSymmetryReason,
    cameraVideoRef,
    startVoice,
    stopVoice,
  };
}
