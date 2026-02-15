import { useEffect, useRef, useState } from "react";
import { Conversation } from "@elevenlabs/client";

import { API_BASE } from "./lib/api.js";
import * as Auth from "./lib/auth.js";
import AuthPanel from "./Auth.jsx";

const apiBase = API_BASE;

const cameraDurationMs = 10000;

const statusColor = {
  Green: "bg-emerald-100 text-moss",
  Yellow: "bg-amber-100 text-gold",
  Red: "bg-rose-100 text-rose",
  Error: "bg-rose-100 text-rose",
  neutral: "bg-amber-50 text-stone-600"
};

const completionPhrase = "Thank you for your responses. The screening is now complete. Goodbye.";

const TOKEN_STORAGE_KEY = "guardian_checkin.jwt";

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

const concatInt16 = (chunks) => {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Int16Array(total);
  let offset = 0;
  chunks.forEach((chunk) => {
    out.set(chunk, offset);
    offset += chunk.length;
  });
  return out;
};

const pcm16ToWavBlob = (pcm16, sampleRate = 16000) => {
  const buffer = new ArrayBuffer(44 + pcm16.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i += 1) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + pcm16.length * 2, true);
  writeString(8, "WAVE");

  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);

  writeString(36, "data");
  view.setUint32(40, pcm16.length * 2, true);

  let offset = 44;
  for (let i = 0; i < pcm16.length; i += 1) {
    view.setInt16(offset, pcm16[i], true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
};

const normalizeAnswer = (questionIndex, text) => {
  if (!text) return null;
  const textLower = text.toLowerCase();
  const yesTerms = ["yes", "yeah", "yep", "affirmative", "true"];
  const noTerms = ["no", "nope", "negative", "false"];

  if (yesTerms.some((term) => textLower.includes(term))) return true;
  if (noTerms.some((term) => textLower.includes(term))) return false;

  if (questionIndex === 0) {
    const positiveTerms = ["good", "fine", "okay", "ok", "well", "great", "better"];
    const negativeTerms = ["bad", "not good", "sick", "unwell", "awful", "worse"];
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

  const sessionRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const cameraVideoRef = useRef(null);
  const checkinIdRef = useRef(null);
  const isSessionOpenRef = useRef(false);
  const lastAudioAtRef = useRef(null);
  const heardAudioRef = useRef(false);
  const completionSentRef = useRef(false);
  const completionTimerRef = useRef(null);
  const voiceStartAtRef = useRef(null);
  const lastUserAudioAtRef = useRef(null);
  const currentQuestionIndexRef = useRef(0);
  const responsesRef = useRef([
    { q: "How are you feeling today?", answer: null, transcript: null },
    { q: "Are you experiencing any dizziness, chest pain, or trouble breathing?", answer: null, transcript: null },
    { q: "Did you take your morning medications?", answer: null, transcript: null }
  ]);
  const conversationRef = useRef(null);
  const sttPathRef = useRef(null);

  // Local mic capture for STT logging while the agent is running.
  const sttStreamRef = useRef(null);
  const sttAudioContextRef = useRef(null);
  const sttProcessorRef = useRef(null);
  const sttCollectingRef = useRef(false);
  const sttSilenceFramesRef = useRef(0);
  const sttChunksRef = useRef([]);
  const sttQueueRef = useRef([]);
  const sttInFlightRef = useRef(false);

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
        if(!email.includes("@gmail.com")) {
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
        body: JSON.stringify({ demo_mode: isDemoMode })
      });

      if (!response.ok) {
        throw new Error("Failed to start check-in");
      }

      const data = await response.json();
      const completeResponse = await fetch(`${apiBase}/checkins/${data.checkin_id}/complete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answers: {
            dizziness: false,
            chest_pain: false,
            trouble_breathing: false
          },
          transcript: "Feeling ok today."
        })
      });

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
    lastAudioAtRef.current = null;
    heardAudioRef.current = false;
    completionSentRef.current = false;
    currentQuestionIndexRef.current = 0;
    responsesRef.current = responsesRef.current.map((item) => ({ ...item, answer: null, transcript: null }));
    sttCollectingRef.current = false;
    sttSilenceFramesRef.current = 0;
    sttChunksRef.current = [];
    sttQueueRef.current = [];
    sttInFlightRef.current = false;
    if (sttProcessorRef.current) {
      sttProcessorRef.current.disconnect();
      sttProcessorRef.current = null;
    }
    if (sttStreamRef.current) {
      sttStreamRef.current.getTracks().forEach((track) => track.stop());
      sttStreamRef.current = null;
    }
    if (sttAudioContextRef.current) {
      await sttAudioContextRef.current.close().catch(() => {});
      sttAudioContextRef.current = null;
    }
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    if (cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = null;
    }
  };

  const stopVoice = async () => {
    setVoiceStatus("Stopping...");
    isSessionOpenRef.current = false;
    if (conversationRef.current) {
      try {
        await conversationRef.current.endSession();
      } catch {
        // ignore
      }
      conversationRef.current = null;
    }
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
      responses: responsesRef.current
    };

    await fetch(`${apiBase}/screenings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(screeningData)
    });
  };

  
  const captureCameraClip = async () => {
    setCameraStatus("Recording 10s...");
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraStreamRef.current = stream;
    if (cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = stream;
    }
    const preferredType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
      ? "video/webm;codecs=vp8"
      : "video/webm";
    const recorder = new MediaRecorder(stream, preferredType ? { mimeType: preferredType } : undefined);
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
    formData.append("metadata", JSON.stringify({ duration_ms: cameraDurationMs }));
    const response = await fetch(`${apiBase}/checkins/${checkinId}/upload`, {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error("Failed to upload camera clip");
    }
    setCameraStatus("Uploaded");
  };

  const startVoice = async () => {
    if (isVoiceLive) return;

    setIsVoiceLive(true);
    setVoiceStatus("Connecting...");
    setVoiceLog([]);
    sttPathRef.current = null;

    try {
      const checkinResponse = await fetch(`${apiBase}/checkins/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ demo_mode: isDemoMode, senior_id: "demo-senior" })
      });
      if (!checkinResponse.ok) throw new Error("Failed to start check-in");
      const checkinData = await checkinResponse.json();
      checkinIdRef.current = checkinData.checkin_id;

      // Fetch a signed URL from our backend (keeps ELEVENLABS_API_KEY off the client).
      const signedUrlResp = await fetch(`${apiBase}/elevenlabs/signed-url`);
      const signedData = await signedUrlResp.json().catch(() => null);
      if (!signedUrlResp.ok) {
        const message = signedData?.detail || signedData?.message || "Failed to fetch ElevenLabs signed URL";
        throw new Error(message);
      }

      setVoiceStatus("Connecting to ElevenLabs...");
      voiceStartAtRef.current = Date.now();

      const conversation = await Conversation.startSession({
        signedUrl: signedData.signed_url,
        connectionType: "websocket",
        onConnect: () => {
          isSessionOpenRef.current = true;
          setVoiceStatus("Listening...");
        },
        onDisconnect: () => {
          isSessionOpenRef.current = false;
          setVoiceStatus("Closed");
        },
        onError: (error) => {
          setVoiceStatus("Error");
          setVoiceLog((prev) => [...prev, `[ElevenLabs error] ${error?.message || String(error)}`]);
        },
        onMessage: (message) => {
          const text = (typeof message === "string" ? message : message?.text || message?.message || "").trim();
          if (!text) return;
          heardAudioRef.current = true;
          setVoiceLog((prev) => [...prev, text]);
          if (text.includes(completionPhrase)) {
            handleCompletion().finally(() => stopVoice());
          }

        }
      });

      conversationRef.current = conversation;

      // In parallel with the agent session, capture mic audio and send utterances to ElevenLabs STT
      // so we have a JSON log of what the user said.
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        sttStreamRef.current = stream;

        const audioContext = new AudioContext();
        sttAudioContextRef.current = audioContext;
        if (audioContext.state === "suspended") {
          await audioContext.resume();
        }

        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);

        const drainQueue = async () => {
          if (sttInFlightRef.current) return;
          const next = sttQueueRef.current.shift();
          if (!next) return;
          sttInFlightRef.current = true;

          try {
            const wavBlob = pcm16ToWavBlob(next, 16000);
            const form = new FormData();
            form.append("file", wavBlob, "utterance.wav");
            form.append("model_id", "scribe_v2");
            form.append("session_id", checkinIdRef.current);
            form.append("role", "user");
            form.append("user_email", authUser?.email || "");
            form.append("checkin_id", checkinIdRef.current);

            const res = await fetch(`${apiBase}/stt/elevenlabs/log`, { method: "POST", body: form });
            const data = await res.json().catch(() => null);
            if (!res.ok) {
              const message = (data && (data.detail || data.message)) || `${res.status} ${res.statusText}`;
              throw new Error(message);
            }

            const text = (data?.text || "").trim();
            const path = data?.path || "";
            if (text) {
              setVoiceLog((prev) => [...prev, `[You] ${text}`]);
            }
            if (path && !sttPathRef.current) {
              sttPathRef.current = path;
              setVoiceLog((prev) => [...prev, `Saved STT JSON: ${path}`]);
            }
          } catch (err) {
            setVoiceLog((prev) => [...prev, `[STT error] ${err?.message || "Transcription failed"}`]);
          } finally {
            sttInFlightRef.current = false;
            // Continue draining if more utterances queued.
            if (sttQueueRef.current.length) {
              drainQueue();
            }
          }
        };

        processor.onaudioprocess = (event) => {
          if (!isSessionOpenRef.current) return;
          const input = event.inputBuffer.getChannelData(0);

          let rms = 0;
          for (let i = 0; i < input.length; i += 1) {
            rms += input[i] * input[i];
          }
          rms = Math.sqrt(rms / input.length);

          const speakingNow = rms > 0.01;
          const resampled = resampleTo16k(input, audioContext.sampleRate);
          const pcm16 = floatToInt16(resampled);

          if (speakingNow) {
            if (!sttCollectingRef.current) {
              sttCollectingRef.current = true;
              sttSilenceFramesRef.current = 0;
              sttChunksRef.current = [];
            }
            sttChunksRef.current.push(pcm16);
          } else if (sttCollectingRef.current) {
            sttSilenceFramesRef.current += 1;
            if (sttSilenceFramesRef.current >= 3) {
              sttCollectingRef.current = false;
              sttSilenceFramesRef.current = 0;
              const merged = concatInt16(sttChunksRef.current);
              sttChunksRef.current = [];

              // Skip tiny clips (< ~350ms at 16k).
              if (merged.length >= 16000 * 0.35) {
                sttQueueRef.current.push(merged);
                drainQueue();
              }
            }
          }
        };

        source.connect(processor);
        processor.connect(audioContext.destination);
        sttProcessorRef.current = processor;
      } catch (err) {
        // Agent can still run even if STT logging fails.
        setVoiceLog((prev) => [...prev, `[STT setup error] ${err?.message || "Unable to start STT logging"}`]);
      }

      // Camera clip is still handled by our app (separate from the voice agent).
      try {
        const videoBlob = await captureCameraClip();
        await uploadCameraClip(checkinIdRef.current, videoBlob);
      } catch (error) {
        setCameraStatus("Error");
        setVoiceLog((prev) => [...prev, error?.message || "Camera capture failed."]);
      }

      // Hint to the agent that camera capture is complete.
      try {
        await conversation.sendUserMessage("Camera done. Begin the screening questions now.");
      } catch {
        // ignore
      }
    } catch (error) {
      setVoiceStatus("Error");
      setVoiceLog((prev) => [...prev, error?.message || "Voice setup failed."]);
      setIsVoiceLive(false);
    }
  };

  const chipClass = statusColor[status] || statusColor.neutral;
  const chipText = status || "—";

  if (!isAuthed) {
    return (
      <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#f6efe5_0%,_#f4f0e8_40%,_#f8f2ed_100%)] text-ink">
        <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col justify-center gap-8 px-6 pb-16 pt-12 sm:px-8">
          <header className="text-center">
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-amber-700">Guardian Check-In</p>
            <h1 className="mt-3 text-4xl font-semibold sm:text-5xl">Sign in to continue</h1>
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

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#f6efe5_0%,_#f4f0e8_40%,_#f8f2ed_100%)] text-ink">
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-6 pb-16 pt-12 sm:px-8">

        <div className="flex items-center justify-between rounded-full border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-stone-700">
          <span>Logged in as <span className="font-semibold">{authUser?.firstName} {authUser?.lastName}</span></span>
          <button
            onClick={logout}
            className="rounded-full bg-stone-900 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-800"
          >
            Log out
          </button>
        </div>
        
        <header className="rounded-[28px] border border-amber-100 bg-amber-50/80 p-8 shadow-hero backdrop-blur">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-amber-700">Guardian Check-In</p>
          <h1 className="mt-3 text-4xl font-semibold sm:text-5xl">Daily health check-in</h1>
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
          <p className="mt-2 text-stone-600">{status ? `Status: ${status}` : "Not started"}</p>
          <div className={`mt-3 inline-flex items-center rounded-full px-4 py-2 text-sm font-semibold ${chipClass}`}>
            {chipText}
          </div>
          <p className="mt-3 text-sm text-stone-600">{reason}</p>
        </section>

        <section className="rounded-2xl border border-amber-100 bg-white p-6 shadow-card">
          <div className="flex flex-col gap-6 lg:flex-row">
            <div className="w-full lg:w-1/2">
              <h3 className="text-lg font-semibold">Camera + Voice Check-In</h3>
              <p className="mt-2 text-sm text-stone-600">
                {isVoiceLive ? `Voice status: ${voiceStatus}` : "Start the live voice assistant to begin."}
              </p>
              <p className="mt-1 text-xs text-stone-500">Camera status: {cameraStatus}</p>
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
                {voiceLog.length === 0 ? "No messages yet." : voiceLog.join("\n\n")}
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
                The camera preview appears while the 10s recording is in progress.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
