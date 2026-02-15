import "./style.css";

const app = document.querySelector("#app");

app.innerHTML = `
  <main class="page">
    <header class="hero">
      <p class="eyebrow">Guardian Check-In</p>
      <h1>Daily health check-in</h1>
      <p class="subhead">Quick camera + voice Q&A, then a simple Green/Yellow/Red result.</p>
      <div class="actions">
        <button id="start-checkin" class="primary">Start Check-In</button>
        <label class="toggle">
          <input id="demo-mode" type="checkbox" checked />
          Demo mode
        </label>
      </div>
    </header>

    <section class="card">
      <h2>Status</h2>
      <p id="status-text">Not started</p>
      <div id="result-chip" class="chip neutral">—</div>
      <p id="reason-text" class="muted">Run a check-in to see triage output.</p>
    </section>

    <section class="grid">
      <div class="card">
        <h3>Camera</h3>
        <p class="muted">Capture placeholder for MVP.</p>
        <button class="secondary">Simulate capture</button>
      </div>
      <div class="card">
        <h3>Voice Q&A</h3>
        <p class="muted">ElevenLabs TTS + transcript placeholder.</p>
        <button class="secondary">Simulate Q&A</button>
      </div>
    </section>
  </main>
`;

const statusText = document.querySelector("#status-text");
const reasonText = document.querySelector("#reason-text");
const resultChip = document.querySelector("#result-chip");

const apiBase = "http://localhost:8000";

const setResult = (status, reason) => {
  statusText.textContent = status ? `Status: ${status}` : "Not started";
  reasonText.textContent = reason || "Run a check-in to see triage output.";
  resultChip.textContent = status || "—";
  resultChip.className = `chip ${status ? status.toLowerCase() : "neutral"}`;
};

document.querySelector("#start-checkin").addEventListener("click", async () => {
  setResult("Starting", "Creating a new check-in...");

  try {
    const demoMode = document.querySelector("#demo-mode").checked;
    const response = await fetch(`${apiBase}/checkins/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ demo_mode: demoMode })
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
    setResult(result.triage_status, result.triage_reasons.join("; "));
  } catch (error) {
    setResult("Error", error.message || "Something went wrong.");
  }
});
