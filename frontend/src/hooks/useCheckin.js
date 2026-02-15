import { useRef, useState } from "react";
import { GoogleGenAI, Modality } from "@google/genai";

import { API_BASE } from "../lib/api.js";
import {
  createAudioBuffer,
  decodeBase64ToInt16,
  encodeToBase64,
  resampleTo16k,
  floatToInt16,
} from "../lib/audio.js";
import {
  modelName,
  systemInstruction,
  completionPhrase,
  cameraDurationMs,
  normalizeAnswer,
  INITIAL_RESPONSES,
  parseSymptomTranscript,
  buildTranscript,
} from "../lib/screening.js";

const apiBase = API_BASE;
const FACE_CAPTURE_BUFFER_MS = 5000;

export default function useCheckin(authUser, authToken) {
  const [status, setStatus] = useState(null);
  const [reason, setReason] = useState("Run a check-in to see triage output.");
  const [isDemoMode, setIsDemoMode] = useState(true);
  const [voiceStatus, setVoiceStatus] = useState("Idle");
  const [voiceLog, setVoiceLog] = useState([]);
  const [isVoiceLive, setIsVoiceLive] = useState(false);
  const [isCheckinComplete, setIsCheckinComplete] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("Idle");
  const [facialSymmetryStatus, setFacialSymmetryStatus] = useState("Not run");
  const [facialSymmetryReason, setFacialSymmetryReason] = useState(
    "Facial symmetry results will appear after camera upload.",
  );

  const sessionRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const micStreamRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const cameraVideoRef = useRef(null);
  const checkinIdRef = useRef(null);
  const isSessionOpenRef = useRef(false);
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const lastAudioAtRef = useRef(null);
  const heardAudioRef = useRef(false);
  const completionSentRef = useRef(false);
  const completionTimerRef = useRef(null);
  const startupAudioNudgeTimerRef = useRef(null);
  const completionPromptedRef = useRef(false);
  const completionPromptAtRef = useRef(null);
  const voiceStartAtRef = useRef(null);
  const lastUserAudioAtRef = useRef(null);
  const lastAiAudioAtRef = useRef(null);
  const lastAiBurstAtRef = useRef(null);
  const aiTurnCountRef = useRef(0);
  const userSpeakingRef = useRef(false);
  const userSpeechStartRef = useRef(null);
  const currentQuestionIndexRef = useRef(0);
  const responsesRef = useRef(INITIAL_RESPONSES.map((item) => ({ ...item })));
  const recognitionRef = useRef(null);

  const getResponseErrorMessage = async (response, fallbackMessage) => {
    try {
      const payload = await response.json();
      const detail = payload?.detail;
      if (typeof detail === "string" && detail.trim()) {
        return detail;
      }
      if (
        detail &&
        typeof detail === "object" &&
        typeof detail.message === "string"
      ) {
        return detail.message;
      }
      if (typeof payload?.message === "string" && payload.message.trim()) {
        return payload.message;
      }
    } catch {
      // Ignore parse errors and fall back to default message.
    }
    return fallbackMessage;
  };

  const startCheckin = async () => {
    setStatus("Starting");
    setReason("Creating a new check-in...");

    try {
      const response = await fetch(`${apiBase}/checkins/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error("Failed to start check-in");
      }

      const data = await response.json();
      const completeResponse = await fetch(
        `${apiBase}/checkins/${data.checkin_id}/complete`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
          body: JSON.stringify({
            answers: {
              dizziness: false,
              chest_pain: false,
              trouble_breathing: false,
            },
            transcript: "Feeling ok today.",
          }),
        },
      );

      if (!completeResponse.ok) {
        throw new Error("Failed to complete check-in");
      }

      const result = await completeResponse.json();
      setStatus(result.triage_status);
      setReason(result.triage_reasons.join("; "));
    } catch (error) {
      setStatus("Error");
      setReason(error?.message || "Something went wrong.");
    }
  };

  const cleanupAudio = async () => {
    setIsVoiceLive(false);
    if (completionTimerRef.current) {
      clearInterval(completionTimerRef.current);
      completionTimerRef.current = null;
    }
    if (startupAudioNudgeTimerRef.current) {
      clearTimeout(startupAudioNudgeTimerRef.current);
      startupAudioNudgeTimerRef.current = null;
    }
    lastAudioAtRef.current = null;
    lastAiAudioAtRef.current = null;
    lastAiBurstAtRef.current = null;
    heardAudioRef.current = false;
    completionSentRef.current = false;
    completionPromptedRef.current = false;
    completionPromptAtRef.current = null;
    aiTurnCountRef.current = 0;
    userSpeakingRef.current = false;
    userSpeechStartRef.current = null;
    currentQuestionIndexRef.current = 0;
    responsesRef.current = responsesRef.current.map((item) => ({
      ...item,
      answer: null,
      transcript: null,
    }));
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    if (cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = null;
    }
    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }
  };

  const stopVoice = async () => {
    setVoiceStatus("Stopping...");
    isSessionOpenRef.current = false;
    await cleanupAudio();
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    setVoiceStatus("Idle");
  };

  const handleCompletion = async () => {
    const responses = responsesRef.current;
    const screeningData = {
      session_id: `screening_${Date.now()}`,
      timestamp: new Date().toISOString(),
      checkin_id: checkinIdRef.current,
      responses,
    };

    const screeningResponse = await fetch(`${apiBase}/screenings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
      },
      body: JSON.stringify(screeningData),
    });
    if (!screeningResponse.ok) {
      setVoiceLog((prev) => [
        ...prev,
        "Warning: Screening data could not be stored.",
      ]);
    }

    if (!checkinIdRef.current) return;

    const symptomResponse = responses[1] || {};
    const symptomTranscript = symptomResponse.transcript || "";
    const parsedSymptoms = parseSymptomTranscript(symptomTranscript);
    const symptomAnswer = symptomResponse.answer;

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

    const completeResponse = await fetch(
      `${apiBase}/checkins/${checkinIdRef.current}/complete`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({ answers, transcript }),
      },
    );
    if (!completeResponse.ok) {
      const completionError = await getResponseErrorMessage(
        completeResponse,
        "Failed to complete check-in",
      );
      setVoiceLog((prev) => [
        ...prev,
        `Completion validation failed: ${completionError}`,
      ]);

      const autoCompleteResponse = await fetch(
        `${apiBase}/checkins/${checkinIdRef.current}/auto-complete`,
        {
          method: "POST",
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        },
      );

      if (!autoCompleteResponse.ok) {
        const autoCompleteError = await getResponseErrorMessage(
          autoCompleteResponse,
          "Auto-complete fallback failed",
        );
        throw new Error(autoCompleteError);
      }

      const detailResponse = await fetch(
        `${apiBase}/checkins/${checkinIdRef.current}`,
        {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        },
      );

      if (detailResponse.ok) {
        const detail = await detailResponse.json();
        const triageReasons = Array.isArray(detail?.triage_reasons)
          ? detail.triage_reasons
          : [];
        setStatus(detail?.triage_status || "Completed");
        setReason(
          triageReasons.length
            ? triageReasons.join("; ")
            : "Check-in completed successfully.",
        );
      } else {
        setStatus("Completed");
        setReason("Check-in completed successfully.");
      }
      setIsCheckinComplete(true);
      return null;
    }
    const result = await completeResponse.json();
    setStatus(result?.triage_status || "Completed");
    setReason(
      Array.isArray(result?.triage_reasons) && result.triage_reasons.length
        ? result.triage_reasons.join("; ")
        : "Check-in completed successfully.",
    );
    setIsCheckinComplete(true);
    return result;
  };

  const maybePromptCompletion = (session) => {
    if (!session || completionPromptedRef.current) return;
    completionPromptedRef.current = true;
    completionPromptAtRef.current = Date.now();
    setVoiceLog((prev) => [...prev, "Requesting final closing message..."]);
    session.sendRealtimeInput({
      text: `All screening questions are complete. Say exactly this sentence now: "${completionPhrase}"`,
    });
  };

  const captureCameraClip = async () => {
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

    return await new Promise((resolve, reject) => {
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      recorder.onerror = (event) => {
        setCameraStatus("Error");
        reject(event.error || new Error("Camera recording failed"));
      };
      recorder.onstop = () => {
        if (cameraStreamRef.current) {
          cameraStreamRef.current.getTracks().forEach((track) => track.stop());
          cameraStreamRef.current = null;
        }
        if (cameraVideoRef.current) {
          cameraVideoRef.current.srcObject = null;
        }
        setCameraStatus("Recorded");
        resolve(new Blob(chunks, { type: recorder.mimeType || "video/webm" }));
      };
      recorder.start();
      setTimeout(() => recorder.stop(), cameraDurationMs);
    });
  };

  const uploadCameraClip = async (checkinId, videoBlob) => {
    setCameraStatus("Uploading...");
    const formData = new FormData();
    formData.append("video", videoBlob, "checkin.webm");
    formData.append(
      "metadata",
      JSON.stringify({ duration_ms: cameraDurationMs }),
    );
    const response = await fetch(`${apiBase}/checkins/${checkinId}/upload`, {
      method: "POST",
      headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
      body: formData,
    });
    if (!response.ok) {
      let detail = "Failed to upload camera clip";
      try {
        const payload = await response.json();
        if (payload?.detail) detail = String(payload.detail);
      } catch {
        // Keep default detail when response body is not JSON.
      }
      throw new Error(detail);
    }
    const data = await response.json();
    if (!data?.facial_symmetry) {
      const detailResponse = await fetch(`${apiBase}/checkins/${checkinId}`, {
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
      });
      if (detailResponse.ok) {
        const detailData = await detailResponse.json();
        data.facial_symmetry = detailData?.facial_symmetry ?? null;
      }
    }
    setCameraStatus("Uploaded");
    return data;
  };

  const playQueuedAudio = () => {
    if (isPlayingRef.current) return;
    const audioContext = audioContextRef.current;
    if (!audioContext || audioQueueRef.current.length === 0) return;

    const { pcm, sampleRate } = audioQueueRef.current.shift();
    isPlayingRef.current = true;
    const buffer = createAudioBuffer(audioContext, pcm, sampleRate);
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.onended = () => {
      isPlayingRef.current = false;
      playQueuedAudio();
    };
    source.start();
  };

  const startVoice = async () => {
    if (isVoiceLive) return;

    setIsCheckinComplete(false);
    let resolveFirstPromptAudio = null;
    let firstPromptAudioStarted = false;
    const firstPromptAudioPromise = new Promise((resolve) => {
      resolveFirstPromptAudio = resolve;
    });

    setIsVoiceLive(true);
    setVoiceStatus("Connecting...");
    setVoiceLog([]);
    setFacialSymmetryStatus("Pending");
    setFacialSymmetryReason("Camera capture in progress...");

    try {
      const checkinResponse = await fetch(`${apiBase}/checkins/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({}),
      });
      if (!checkinResponse.ok) {
        throw new Error("Failed to start check-in");
      }
      const checkinData = await checkinResponse.json();
      checkinIdRef.current = checkinData.checkin_id;

      const tokenResponse = await fetch(`${apiBase}/auth/ephemeral`, {
        method: "POST",
      });
      if (!tokenResponse.ok) {
        throw new Error("Failed to fetch ephemeral token");
      }
      const tokenData = await tokenResponse.json();

      const ai = new GoogleGenAI({
        apiKey: tokenData.token,
        httpOptions: { apiVersion: "v1alpha" },
      });
      const session = await ai.live.connect({
        model: modelName,
        httpOptions: { apiVersion: "v1alpha" },
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction,
        },
        callbacks: {
          onmessage: (message) => {
            if (message.serverContent?.modelTurn?.parts) {
              message.serverContent.modelTurn.parts.forEach((part) => {
                if (part.inlineData?.data) {
                  const audioContext = audioContextRef.current;
                  if (!audioContext) return;
                  const pcm = decodeBase64ToInt16(part.inlineData.data);
                  if (!firstPromptAudioStarted) {
                    firstPromptAudioStarted = true;
                    resolveFirstPromptAudio?.();
                  }
                  lastAudioAtRef.current = Date.now();
                  lastAiAudioAtRef.current = Date.now();
                  if (
                    !lastAiBurstAtRef.current ||
                    Date.now() - lastAiBurstAtRef.current > 1000
                  ) {
                    aiTurnCountRef.current += 1;
                    const questionIndex = Math.max(
                      0,
                      Math.min(
                        aiTurnCountRef.current - 2,
                        responsesRef.current.length - 1,
                      ),
                    );
                    currentQuestionIndexRef.current = questionIndex;
                    console.log(
                      "[Live AI] Turn",
                      aiTurnCountRef.current,
                      "Question index",
                      questionIndex,
                    );
                  }
                  lastAiBurstAtRef.current = Date.now();
                  heardAudioRef.current = true;
                  audioQueueRef.current.push({ pcm, sampleRate: 24000 });
                  playQueuedAudio();
                }

                if (part.text) {
                  console.log("[Live AI]", part.text);
                  setVoiceLog((prev) => [...prev, part.text]);
                  if (
                    !completionSentRef.current &&
                    part.text
                      .toLowerCase()
                      .includes(completionPhrase.toLowerCase())
                  ) {
                    completionSentRef.current = true;
                    console.log(
                      "[Live AI] Completion phrase detected. Posting screening payload.",
                    );
                    handleCompletion()
                      .catch((error) => {
                        setVoiceStatus("Error");
                        setVoiceLog((prev) => [
                          ...prev,
                          error?.message || "Check-in completion failed.",
                        ]);
                      })
                      .finally(() => {
                        stopVoice();
                      });
                  }
                }
              });
            }
          },
          onerror: (e) => {
            setVoiceStatus("Error");
            setVoiceLog((prev) => [...prev, `Error: ${e.message}`]);
          },
          onclose: (event) => {
            isSessionOpenRef.current = false;
            cleanupAudio();
            setVoiceStatus("Closed");
            if (event?.reason) {
              setVoiceLog((prev) => [...prev, `Closed: ${event.reason}`]);
            }
          },
        },
      });

      sessionRef.current = session;
      isSessionOpenRef.current = true;
      setVoiceStatus("Listening...");
      voiceStartAtRef.current = Date.now();

      if (!audioContextRef.current) {
        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;
        if (audioContext.state === "suspended") {
          await audioContext.resume();
        }
      }

      session.sendRealtimeInput({
        text: "Tell the user now: 'We are collecting your face data for the next 10 seconds. Please keep your face centered and still.' Then wait for my camera done signal before asking screening questions.",
      });
      setVoiceLog((prev) => [
        ...prev,
        "Collecting face data for 10 seconds. Please keep your face centered and still.",
      ]);
      startupAudioNudgeTimerRef.current = setTimeout(() => {
        if (!isSessionOpenRef.current || sessionRef.current !== session) return;
        if (heardAudioRef.current) return;
        session.sendRealtimeInput({
          text: "Respond now in spoken audio.",
        });
      }, 3500);

      try {
        await Promise.race([
          firstPromptAudioPromise,
          new Promise((resolve) => {
            setTimeout(resolve, 5000);
          }),
        ]);
        await new Promise((resolve) => {
          setTimeout(resolve, FACE_CAPTURE_BUFFER_MS);
        });
        if (!isSessionOpenRef.current || sessionRef.current !== session) {
          throw new Error("Session closed before camera capture.");
        }
        const videoBlob = await captureCameraClip();
        const uploadResult = await uploadCameraClip(
          checkinIdRef.current,
          videoBlob,
        );
        const facialSymmetry = uploadResult?.facial_symmetry;
        if (facialSymmetry?.status) {
          const prefix =
            facialSymmetry.status === "ERROR"
              ? "Facial symmetry unavailable"
              : `Facial symmetry ${facialSymmetry.status}`;
          const detail = facialSymmetry.reason || "No details returned.";
          setFacialSymmetryStatus(facialSymmetry.status);
          setFacialSymmetryReason(detail);
          setVoiceLog((prev) => [...prev, `${prefix}: ${detail}`]);
        } else {
          setFacialSymmetryStatus("Missing");
          setFacialSymmetryReason(
            "No facial symmetry payload was returned from backend.",
          );
          setVoiceLog((prev) => [
            ...prev,
            "Facial symmetry result missing from backend response.",
          ]);
        }
      } catch (error) {
        setCameraStatus("Error");
        setFacialSymmetryStatus("Error");
        setFacialSymmetryReason(
          error?.message || "Camera capture/upload failed.",
        );
        setVoiceLog((prev) => [
          ...prev,
          error?.message || "Camera capture failed.",
        ]);
      }

      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = "en-US";
        recognition.onresult = (event) => {
          const result = event.results[event.results.length - 1];
          if (!result?.isFinal) return;
          const transcript = result[0]?.transcript?.trim();
          if (!transcript) return;
          const idx = currentQuestionIndexRef.current;
          const response = responsesRef.current[idx];
          responsesRef.current[idx] = {
            ...response,
            transcript,
            answer: normalizeAnswer(idx, transcript),
          };
          console.log("[User] Transcript", transcript);
          const answeredCount = responsesRef.current.filter(
            (item) => item.transcript,
          ).length;
          if (answeredCount >= responsesRef.current.length) {
            maybePromptCompletion(session);
          }
        };
        recognition.onerror = (event) => {
          console.warn("[User] Speech recognition error", event?.error);
        };
        recognition.start();
        recognitionRef.current = recognition;
      } else {
        console.warn(
          "[User] Speech recognition not supported in this browser.",
        );
      }

      completionTimerRef.current = setInterval(() => {
        if (!heardAudioRef.current || completionSentRef.current) {
          return;
        }
        const lastAudioAt = lastAudioAtRef.current;
        if (!lastAudioAt) return;
        const idleMs = Date.now() - lastAudioAt;
        const startedAt = voiceStartAtRef.current ?? 0;
        const sessionMs = Date.now() - startedAt;
        const lastAiAt = lastAiAudioAtRef.current;
        const aiIdleMs = lastAiAt ? Date.now() - lastAiAt : 0;
        const answeredCount = responsesRef.current.filter(
          (item) => item.transcript,
        ).length;

        if (
          answeredCount >= responsesRef.current.length &&
          !completionPromptedRef.current &&
          aiIdleMs >= 1500
        ) {
          maybePromptCompletion(session);
          return;
        }

        if (completionPromptedRef.current) {
          const promptedAt = completionPromptAtRef.current ?? Date.now();
          const promptAgeMs = Date.now() - promptedAt;
          if (aiIdleMs >= 4500 && promptAgeMs >= 4500) {
            completionSentRef.current = true;
            console.log(
              "[Live AI] Final message window elapsed. Posting screening payload.",
            );
            handleCompletion()
              .catch((error) => {
                setVoiceStatus("Error");
                setVoiceLog((prev) => [
                  ...prev,
                  error?.message || "Check-in completion failed.",
                ]);
              })
              .finally(() => {
                stopVoice();
              });
          }
          return;
        }

        if (
          aiTurnCountRef.current >= 4 &&
          idleMs >= 6000 &&
          sessionMs >= 18000
        ) {
          maybePromptCompletion(session);
        }
      }, 500);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        micStreamRef.current = stream;

        const audioContext = audioContextRef.current;
        if (!audioContext) {
          throw new Error("Audio context missing");
        }

        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (event) => {
          if (!isSessionOpenRef.current || sessionRef.current !== session) {
            return;
          }
          const input = event.inputBuffer.getChannelData(0);
          let rms = 0;
          for (let i = 0; i < input.length; i += 1) {
            rms += input[i] * input[i];
          }
          rms = Math.sqrt(rms / input.length);
          if (rms > 0.01) {
            lastUserAudioAtRef.current = Date.now();
            if (!userSpeakingRef.current) {
              userSpeakingRef.current = true;
              userSpeechStartRef.current = Date.now();
              console.log("[User] Speech started");
            }
          } else if (userSpeakingRef.current) {
            userSpeakingRef.current = false;
            const startedAt = userSpeechStartRef.current ?? Date.now();
            const durationMs = Date.now() - startedAt;
            console.log(
              "[User] Speech ended",
              `${Math.round(durationMs / 100) / 10}s`,
            );
            userSpeechStartRef.current = null;
          }
          const resampled = resampleTo16k(input, audioContext.sampleRate);
          const pcm16 = floatToInt16(resampled);
          const base64Audio = encodeToBase64(pcm16);
          session.sendRealtimeInput({
            audio: {
              data: base64Audio,
              mimeType: "audio/pcm;rate=16000",
            },
          });
        };

        source.connect(processor);
        processor.connect(audioContext.destination);
        processorRef.current = processor;

        setTimeout(() => {
          if (!isSessionOpenRef.current || sessionRef.current !== session) {
            return;
          }
          session.sendRealtimeInput({
            text: "Camera done. Begin the screening questions now.",
          });
        }, 200);
      } catch (error) {
        setVoiceStatus("Error");
        setVoiceLog((prev) => [...prev, error?.message || "Mic setup failed."]);
        stopVoice();
      }
    } catch (error) {
      setVoiceStatus("Error");
      setVoiceLog((prev) => [...prev, error?.message || "Voice setup failed."]);
      setIsVoiceLive(false);
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
    isCheckinComplete,
    cameraStatus,
    facialSymmetryStatus,
    facialSymmetryReason,
    cameraVideoRef,
    startCheckin,
    startVoice,
    stopVoice,
  };
}
