import { useEffect, useMemo, useRef, useState } from "react";
import { GoogleGenAI, Modality } from "@google/genai";

import { API_BASE, apiFetch } from "./lib/api.js";
import * as Auth from "./lib/auth.js";
import AuthPanel from "./Auth.jsx";

const apiBase = API_BASE;
const modelName = "gemini-2.5-flash-native-audio-preview-12-2025";
const systemInstruction = `ROLE: You are a professional Medical Screening Assistant.
OBJECTIVE: Conduct a brief well-being check by asking specific questions.
RULES:
1. First ask the user to look at their camera for 10 seconds and wait for a "camera done" signal.
2. Then  tell them you are about to ask them some questions on their general wellbeing.First ask: "How are you feeling today?"
3. Follow up with: "Are you experiencing any dizziness, chest pain, or trouble breathing?"
4. Finally ask: "Did you take your morning medications?"
5. STRICT: Only ask these questions. If the user tries to change the subject, politely redirect them back to the screening.
6. TERMINATION: Once all questions are answered, say exactly: "Thank you for your responses. The screening is now complete. Goodbye."`;

const cameraDurationMs = 10000;

const statusColor = {
  Green: "bg-emerald-100 text-moss",
  Yellow: "bg-amber-100 text-gold",
  Red: "bg-rose-100 text-rose",
  Error: "bg-rose-100 text-rose",
  neutral: "bg-amber-50 text-stone-600",
};

const completionPhrase =
  "Thank you for your responses. The screening is now complete. Goodbye.";

const TOKEN_STORAGE_KEY = "guardian_checkin.jwt";

const createAudioBuffer = (audioContext, pcm16, sampleRate) => {
  const floatData = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i += 1) {
    floatData[i] = Math.max(-1, Math.min(1, pcm16[i] / 32768));
  }
  const buffer = audioContext.createBuffer(1, floatData.length, sampleRate);
  buffer.copyToChannel(floatData, 0);
  return buffer;
};

const decodeBase64ToInt16 = (base64) => {
  const binary = atob(base64);
  const length = binary.length;
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
};

const encodeToBase64 = (int16) => {
  const bytes = new Uint8Array(int16.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

const resampleTo16k = (input, inputRate) => {
  if (inputRate === 16000) {
    return input;
  }
  const ratio = inputRate / 16000;
  const newLength = Math.round(input.length / ratio);
  const output = new Float32Array(newLength);
  for (let i = 0; i < newLength; i += 1) {
    const position = i * ratio;
    const leftIndex = Math.floor(position);
    const rightIndex = Math.min(leftIndex + 1, input.length - 1);
    const weight = position - leftIndex;
    output[i] = input[leftIndex] * (1 - weight) + input[rightIndex] * weight;
  }
  return output;
};

const floatToInt16 = (floatData) => {
  const output = new Int16Array(floatData.length);
  for (let i = 0; i < floatData.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, floatData[i]));
    output[i] = sample < 0 ? sample * 32768 : sample * 32767;
  }
  return output;
};

const normalizeAnswer = (questionIndex, text) => {
  if (!text) return null;
  const textLower = text.toLowerCase();
  const yesTerms = ["yes", "yeah", "yep", "affirmative", "true"];
  const noTerms = ["no", "nope", "negative", "false"];

  if (yesTerms.some((term) => textLower.includes(term))) return true;
  if (noTerms.some((term) => textLower.includes(term))) return false;

  if (questionIndex === 0) {
    const positiveTerms = [
      "good",
      "fine",
      "okay",
      "ok",
      "well",
      "great",
      "better",
    ];
    const negativeTerms = [
      "bad",
      "not good",
      "sick",
      "unwell",
      "awful",
      "worse",
    ];
    if (positiveTerms.some((term) => textLower.includes(term))) return true;
    if (negativeTerms.some((term) => textLower.includes(term))) return false;
  }

  return null;
};

export default function App() {
  const [authMode, setAuthMode] = useState("login"); // login | register
  const [authFirstName, setAuthFirstName] = useState("");
  const [authLastName, setAuthLastName] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authUser, setAuthUser] = useState(null);
  const [authToken, setAuthToken] = useState(() => {
    try {
      return localStorage.getItem(TOKEN_STORAGE_KEY);
    } catch {
      return null;
    }
  });
  const [authStatus, setAuthStatus] = useState("idle"); // idle | loading
  const [authError, setAuthError] = useState(null);
  const isAuthed = Boolean(authToken && authUser?.email);

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
  const [doctorSeniors, setDoctorSeniors] = useState([]);
  const [doctorStats, setDoctorStats] = useState({
    total_seniors: 0,
    total_checkins: 0,
    green: 0,
    yellow: 0,
    red: 0,
    alerts: 0,
    window_days: 7
  });
  const [doctorLoading, setDoctorLoading] = useState(false);
  const [doctorError, setDoctorError] = useState(null);

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
  const responsesRef = useRef([
    { q: "How are you feeling today?", answer: null, transcript: null },
    {
      q: "Are you experiencing any dizziness, chest pain, or trouble breathing?",
      answer: null,
      transcript: null,
    },
    {
      q: "Did you take your morning medications?",
      answer: null,
      transcript: null,
    },
  ]);
  const recognitionRef = useRef(null);

  const persistToken = (token) => {
    setAuthToken(token);
    try {
      if (token) localStorage.setItem(TOKEN_STORAGE_KEY, token);
      else localStorage.removeItem(TOKEN_STORAGE_KEY);
    } catch {
      // ignore
    }
  };

  const refreshMe = async (token) => {
    if (!token) {
      setAuthUser(null);
      return;
    }
    try {
      const user = await Auth.me({ token });
      setAuthUser(user);
    } catch (err) {
      // Token invalid or backend unavailable: drop token so the UI recovers.
      persistToken(null);
      setAuthUser(null);
    }
  };

  useEffect(() => {
    refreshMe(authToken);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submitAuth = async (event) => {
    event?.preventDefault?.();
    setAuthError(null);
    setAuthStatus("loading");
    const firstName = authFirstName.trim();
    const lastName = authLastName.trim();
    const email = authEmail.trim().toLowerCase();
    const password = authPassword;

    try {
      if (!email || !password) {
        throw new Error("Email and password are required");
      }

      if (authMode === "register") {
        if (!firstName || !lastName) {
          throw new Error("First name and last name are required");
        }
        if (!email.includes("@gmail.com")) {
          throw new Error("Please use a valid Gmail address");
        }
        await Auth.register({ firstName, lastName, email, password });
      }

      const { access_token } = await Auth.login({ email, password });
      persistToken(access_token);
      await refreshMe(access_token);
      setAuthPassword("");
      setAuthFirstName("");
      setAuthLastName("");
    } catch (err) {
      setAuthError(err?.message || "Auth failed");
    } finally {
      setAuthStatus("idle");
    }
  };

  const logout = () => {
    persistToken(null);
    setAuthUser(null);
    setAuthError(null);
  };

  const startCheckin = async () => {
    setStatus("Starting");
    setReason("Creating a new check-in...");

    try {
      const response = await fetch(`${apiBase}/checkins/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ demo_mode: isDemoMode }),
      });

      if (!response.ok) {
        throw new Error("Failed to start check-in");
      }

      const data = await response.json();
      const completeResponse = await fetch(
        `${apiBase}/checkins/${data.checkin_id}/complete`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
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
    const screeningData = {
      session_id: `screening_${Date.now()}`,
      timestamp: new Date().toISOString(),
      senior_id: "demo-senior",
      checkin_id: checkinIdRef.current,
      responses: responsesRef.current,
    };

    await fetch(`${apiBase}/screenings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(screeningData),
    });
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
      const detailResponse = await fetch(`${apiBase}/checkins/${checkinId}`);
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

    setIsVoiceLive(true);
    setVoiceStatus("Connecting...");
    setVoiceLog([]);
    setFacialSymmetryStatus("Pending");
    setFacialSymmetryReason("Camera capture in progress...");

    try {
      const checkinResponse = await fetch(`${apiBase}/checkins/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          demo_mode: isDemoMode,
          senior_id: "demo-senior",
        }),
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
                  if (part.text.includes(completionPhrase)) {
                    console.log(
                      "[Live AI] Completion phrase detected. Posting screening payload.",
                    );
                    handleCompletion().finally(() => {
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
            handleCompletion().finally(() => {
              stopVoice();
            });
          }
          return;
        }

        if (aiTurnCountRef.current >= 4 && idleMs >= 6000 && sessionMs >= 18000) {
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

  const chipClass = statusColor[status] || statusColor.neutral;
  const chipText = status || "—";
  const isDoctor = authUser?.role === "doctor";
  const formattedSeniors = useMemo(() => {
    return doctorSeniors.map((senior) => {
      const lastCheckinAt = senior.lastCheckinAt
        ? new Date(senior.lastCheckinAt).toLocaleString()
        : "No check-in yet";
      const name = `${senior.firstName || ""} ${senior.lastName || ""}`.trim() || "Unknown";
      const triage = senior.triageStatus
        ? `${senior.triageStatus}`.replace(/^[a-z]/, (char) => char.toUpperCase())
        : "—";
      return {
        ...senior,
        displayName: name,
        lastCheckinAt,
        triage
      };
    });
  }, [doctorSeniors]);

  useEffect(() => {
    const loadDoctorDashboard = async () => {
      if (!isDoctor || !authToken) return;
      setDoctorLoading(true);
      setDoctorError(null);
      try {
        const [analytics, seniors] = await Promise.all([
          apiFetch("/dashboard/analytics", { token: authToken }),
          apiFetch("/dashboard/seniors", { token: authToken })
        ]);
        setDoctorStats(analytics);
        setDoctorSeniors(seniors?.seniors || []);
      } catch (err) {
        setDoctorError(err?.message || "Failed to load dashboard data");
      } finally {
        setDoctorLoading(false);
      }
    };

    loadDoctorDashboard();
  }, [authToken, isDoctor]);

  if (!isAuthed) {
    return (
      <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#f6efe5_0%,_#f4f0e8_40%,_#f8f2ed_100%)] text-ink">
        <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col justify-center gap-8 px-6 pb-16 pt-12 sm:px-8">
          <header className="text-center">
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-amber-700">
              Guardian Check-In
            </p>
            <h1 className="mt-3 text-4xl font-semibold sm:text-5xl">
              Sign in to continue
            </h1>
            <p className="mt-3 text-base text-stone-600">
              Access the check-in dashboard after authentication.
            </p>
          </header>
          <AuthPanel
            authStatus={authStatus}
            authToken={authToken}
            authUser={authUser}
            apiBase={apiBase}
            refreshMe={refreshMe}
            logout={logout}
            submitAuth={submitAuth}
            authMode={authMode}
            setAuthMode={setAuthMode}
            authFirstName={authFirstName}
            setAuthFirstName={setAuthFirstName}
            authLastName={authLastName}
            setAuthLastName={setAuthLastName}
            authEmail={authEmail}
            setAuthEmail={setAuthEmail}
            authPassword={authPassword}
            setAuthPassword={setAuthPassword}
            authError={authError}
          />
        </main>
      </div>
    );
  }

  if (isDoctor) {
    return (
      <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#f6efe5_0%,_#f4f0e8_40%,_#f8f2ed_100%)] text-ink">
        <main className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-6 pb-16 pt-12 sm:px-8">
          <div className="flex items-center justify-between rounded-full border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-stone-700">
            <span>
              Doctor dashboard · <span className="font-semibold">{authUser?.firstName} {authUser?.lastName}</span>
            </span>
            <button
              onClick={logout}
              className="rounded-full bg-stone-900 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-800"
            >
              Log out
            </button>
          </div>

          <header className="rounded-[28px] border border-amber-100 bg-amber-50/80 p-8 shadow-hero backdrop-blur">
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-amber-700">Guardian Analytics</p>
            <h1 className="mt-3 text-4xl font-semibold sm:text-5xl">Clinical overview</h1>
            <p className="mt-3 max-w-2xl text-lg text-stone-600">
              Live pulse on seniors, triage distribution, and follow-up priorities.
            </p>
          </header>

          <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-2xl border border-amber-100 bg-white p-6 shadow-card">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">Total seniors</p>
              <p className="mt-3 text-3xl font-semibold text-stone-900">{doctorStats.total_seniors}</p>
              <p className="mt-2 text-sm text-stone-600">Active in the last 7 days</p>
            </div>
            <div className="rounded-2xl border border-emerald-100 bg-emerald-50/60 p-6 shadow-card">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-700">Green status</p>
              <p className="mt-3 text-3xl font-semibold text-emerald-800">{doctorStats.green}</p>
              <p className="mt-2 text-sm text-emerald-700">Stable check-ins</p>
            </div>
            <div className="rounded-2xl border border-amber-100 bg-amber-50/60 p-6 shadow-card">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-700">Yellow alerts</p>
              <p className="mt-3 text-3xl font-semibold text-amber-800">{doctorStats.yellow}</p>
              <p className="mt-2 text-sm text-amber-700">Monitor within 24 hours</p>
            </div>
            <div className="rounded-2xl border border-rose-100 bg-rose-50/60 p-6 shadow-card">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-rose-700">Red alerts</p>
              <p className="mt-3 text-3xl font-semibold text-rose-800">{doctorStats.red}</p>
              <p className="mt-2 text-sm text-rose-700">Immediate follow-up</p>
            </div>
          </section>

          <section className="rounded-2xl border border-amber-100 bg-white p-6 shadow-card">
            <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-xl font-semibold">Senior overview</h2>
                <p className="text-sm text-stone-600">Latest check-ins and context for follow-up.</p>
              </div>
              <div className="rounded-full border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-stone-700">
                {doctorStats.alerts} seniors need attention
              </div>
            </div>
            <div className="mt-6 grid gap-4">
              {doctorLoading ? (
                <div className="rounded-2xl border border-stone-200 bg-stone-50/60 p-5 text-sm text-stone-600">
                  Loading dashboard data...
                </div>
              ) : doctorError ? (
                <div className="rounded-2xl border border-rose-100 bg-rose-50/60 p-5 text-sm text-rose-700">
                  {doctorError}
                </div>
              ) : formattedSeniors.length === 0 ? (
                <div className="rounded-2xl border border-stone-200 bg-stone-50/60 p-5 text-sm text-stone-600">
                  No seniors found yet.
                </div>
              ) : null}
              {formattedSeniors.map((senior) => (
                <div
                  key={senior.id}
                  className="flex flex-col gap-4 rounded-2xl border border-stone-200 bg-stone-50/60 p-5 md:flex-row md:items-center md:justify-between"
                >
                  <div>
                    <p className="text-lg font-semibold text-stone-900">{senior.displayName}</p>
                    <p className="text-sm text-stone-600">Last check-in {senior.lastCheckinAt}</p>
                    <p className="mt-2 text-sm text-stone-600">{senior.email}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className={`rounded-full px-4 py-2 text-xs font-semibold ${
                        statusColor[senior.triage] || statusColor.neutral
                      }`}
                    >
                      {senior.triage}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#f6efe5_0%,_#f4f0e8_40%,_#f8f2ed_100%)] text-ink">
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-6 pb-16 pt-12 sm:px-8">
        <div className="flex items-center justify-between rounded-full border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-stone-700">
          <span>
            Logged in as{" "}
            <span className="font-semibold">
              {authUser?.firstName} {authUser?.lastName}
            </span>
          </span>
          <button
            onClick={logout}
            className="rounded-full bg-stone-900 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-800"
          >
            Log out
          </button>
        </div>

        <header className="rounded-[28px] border border-amber-100 bg-amber-50/80 p-8 shadow-hero backdrop-blur">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-amber-700">
            Guardian Check-In
          </p>
          <h1 className="mt-3 text-4xl font-semibold sm:text-5xl">
            Daily health check-in
          </h1>
          <p className="mt-3 max-w-2xl text-lg text-stone-600">
            Quick camera + voice Q&amp;A, then a simple Green/Yellow/Red result.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-4">
            <button
              className="rounded-full bg-clay px-6 py-3 text-base font-semibold text-white shadow-lg shadow-orange-200/60 transition hover:-translate-y-0.5"
              onClick={startCheckin}
            >
              Start Check-In
            </button>
            <label className="flex items-center gap-2 text-sm text-stone-700">
              <input
                type="checkbox"
                className="h-4 w-4 accent-clay"
                checked={isDemoMode}
                onChange={(event) => setIsDemoMode(event.target.checked)}
              />
              Demo mode
            </label>
          </div>
        </header>

        <section className="rounded-2xl border border-amber-100 bg-white p-6 shadow-card">
          <h2 className="text-xl font-semibold">Status</h2>
          <p className="mt-2 text-stone-600">
            {status ? `Status: ${status}` : "Not started"}
          </p>
          <div
            className={`mt-3 inline-flex items-center rounded-full px-4 py-2 text-sm font-semibold ${chipClass}`}
          >
            {chipText}
          </div>
          <p className="mt-3 text-sm text-stone-600">{reason}</p>
        </section>

        <section className="rounded-2xl border border-amber-100 bg-white p-6 shadow-card">
          <div className="flex flex-col gap-6 lg:flex-row">
            <div className="w-full lg:w-1/2">
              <h3 className="text-lg font-semibold">Camera + Voice Check-In</h3>
              <p className="mt-2 text-sm text-stone-600">
                {isVoiceLive
                  ? `Voice status: ${voiceStatus}`
                  : "Start the live voice assistant to begin."}
              </p>
              <p className="mt-1 text-xs text-stone-500">
                Camera status: {cameraStatus}
              </p>
              <p className="mt-1 text-xs text-stone-500">
                Facial symmetry: {facialSymmetryStatus}
              </p>
              <p className="mt-1 text-xs text-stone-500">
                {facialSymmetryReason}
              </p>
              <div className="mt-4 flex flex-wrap gap-3">
                <button
                  className="rounded-xl border border-amber-200 px-4 py-2 text-sm text-stone-700"
                  onClick={startVoice}
                  disabled={isVoiceLive}
                >
                  Start session
                </button>
                <button
                  className="rounded-xl border border-amber-200 px-4 py-2 text-sm text-stone-700"
                  onClick={stopVoice}
                  disabled={!isVoiceLive}
                >
                  Stop
                </button>
              </div>
              <div className="mt-4 max-h-40 overflow-auto rounded-lg bg-amber-50/60 p-3 text-xs text-stone-700">
                {voiceLog.length === 0
                  ? "No messages yet."
                  : voiceLog.join("\n\n")}
              </div>
            </div>
            <div className="w-full lg:w-1/2">
              <div className="aspect-video overflow-hidden rounded-2xl border border-amber-100 bg-amber-50/50">
                <video
                  ref={cameraVideoRef}
                  autoPlay
                  muted
                  playsInline
                  className="h-full w-full object-cover"
                />
              </div>
              <p className="mt-2 text-xs text-stone-500">
                The camera preview appears while the 10s recording is in
                progress.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
