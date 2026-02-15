import { useState } from "react";
import useDoctorDashboard from "../hooks/useDoctorDashboard.js";
import { apiFetch } from "../lib/api.js";
import { statusColor } from "../lib/screening.js";

const formatDateTime = (value) =>
  value ? new Date(value).toLocaleString() : "Unknown";

const buildTriageCounts = (checkins) =>
  checkins.reduce(
    (acc, checkin) => {
      const status = `${checkin.triage_status || ""}`.toLowerCase();
      if (status === "green") acc.green += 1;
      if (status === "yellow") acc.yellow += 1;
      if (status === "red") acc.red += 1;
      return acc;
    },
    { green: 0, yellow: 0, red: 0 },
  );

const buildSignalCounts = (checkins) =>
  checkins.reduce(
    (acc, checkin) => {
      const signalText = `${checkin.transcript || ""} ${(checkin.triage_reasons || []).join(" ")}`
        .toLowerCase()
        .trim();
      if (signalText.includes("dizz")) acc.dizziness += 1;
      if (signalText.includes("chest")) acc.chest_pain += 1;
      if (signalText.includes("breath")) acc.breathing += 1;
      if (signalText.includes("med") && (signalText.includes("miss") || signalText.includes("not"))) {
        acc.medication_missed += 1;
      }
      return acc;
    },
    { dizziness: 0, chest_pain: 0, breathing: 0, medication_missed: 0 },
  );

const buildReportPayload = ({ senior, checkins, stats, signals }) => ({
  senior_id: senior.id,
  senior_name: senior.displayName,
  senior_email: senior.email,
  overview: {
    total_checkins: stats.total,
    last_checkin_at: stats.lastCheckinAt || null,
    days_since_last_checkin: stats.daysSinceLast,
    triage_counts: stats.triageCounts,
    signal_counts: signals,
  },
  recent_checkins: checkins.slice(0, 8).map((checkin) => ({
    completed_at: checkin.completed_at || null,
    triage_status: checkin.triage_status || null,
    triage_reasons: checkin.triage_reasons || [],
  })),
});

const normalizeTriageLabel = (value) => {
  if (!value) return "—";
  const trimmed = `${value}`.trim();
  if (!trimmed) return "—";
  const lowered = trimmed.toLowerCase();
  if (lowered === "green") return "Green";
  if (lowered === "yellow") return "Yellow";
  if (lowered === "red") return "Red";
  if (lowered === "error") return "Error";
  return trimmed.replace(/^[a-z]/, (char) => char.toUpperCase());
};

export default function DoctorDashboard({ authUser, authToken, logout }) {
  const { doctorStats, formattedSeniors, doctorLoading, doctorError } =
    useDoctorDashboard({ authToken, isDoctor: true });

  const [reportOpen, setReportOpen] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [selectedSenior, setSelectedSenior] = useState(null);

  const closeReport = () => {
    setReportOpen(false);
    setReportLoading(false);
    setReportError(null);
    setReportData(null);
    setSelectedSenior(null);
  };

  const openReport = async (senior) => {
    setSelectedSenior(senior);
    setReportOpen(true);
    setReportLoading(true);
    setReportError(null);
    setReportData(null);

    try {
      const response = await apiFetch(`/seniors/${senior.id}/checkins`, {
        token: authToken,
      });
      const checkins = response?.items || [];

      const triageCounts = buildTriageCounts(checkins);
      const lastCheckinAt = checkins[0]?.completed_at
        ? formatDateTime(checkins[0].completed_at)
        : null;
      const daysSinceLast = checkins[0]?.completed_at
        ? Math.max(
            0,
            Math.floor(
              (Date.now() - new Date(checkins[0].completed_at).getTime()) /
                (1000 * 60 * 60 * 24),
            ),
          )
        : null;
      const signals = buildSignalCounts(checkins);

      const stats = {
        total: checkins.length,
        lastCheckinAt,
        daysSinceLast,
        triageCounts,
      };

      let summaryData = null;
      try {
        const summaryResponse = await apiFetch("/reports/senior-summary", {
          method: "POST",
          token: authToken,
          body: buildReportPayload({ senior, checkins, stats, signals }),
        });
        summaryData = summaryResponse || null;
      } catch (aiError) {
        console.error("[DoctorDashboard] Failed to generate AI summary", aiError);
        summaryData = null;
      }

      setReportData({
        checkins,
        stats,
        signals,
        summaryData,
        generatedAt: new Date().toLocaleString(),
      });
    } catch (err) {
      setReportError(err?.message || "Failed to load report data");
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#f6efe5_0%,_#f4f0e8_40%,_#f8f2ed_100%)] text-ink">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-6 pb-16 pt-12 sm:px-8">
        <div className="flex items-center justify-between rounded-full border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-stone-700">
          <span>
            Doctor dashboard ·{" "}
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
            Guardian Analytics
          </p>
          <h1 className="mt-3 text-4xl font-semibold sm:text-5xl">
            Clinical overview
          </h1>
          <p className="mt-3 max-w-2xl text-lg text-stone-600">
            Live pulse on seniors, triage distribution, and follow-up
            priorities.
          </p>
        </header>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl border border-amber-100 bg-white p-6 shadow-card">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
              Total seniors
            </p>
            <p className="mt-3 text-3xl font-semibold text-stone-900">
              {doctorStats.total_seniors}
            </p>
            <p className="mt-2 text-sm text-stone-600">
              Active in the last 7 days
            </p>
          </div>
          <div className="rounded-2xl border border-emerald-100 bg-emerald-50/60 p-6 shadow-card">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-700">
              Green status
            </p>
            <p className="mt-3 text-3xl font-semibold text-emerald-800">
              {doctorStats.green}
            </p>
            <p className="mt-2 text-sm text-emerald-700">Stable check-ins</p>
          </div>
          <div className="rounded-2xl border border-amber-100 bg-amber-50/60 p-6 shadow-card">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-700">
              Yellow alerts
            </p>
            <p className="mt-3 text-3xl font-semibold text-amber-800">
              {doctorStats.yellow}
            </p>
            <p className="mt-2 text-sm text-amber-700">
              Monitor within 24 hours
            </p>
          </div>
          <div className="rounded-2xl border border-rose-100 bg-red-50/60 p-6 shadow-card">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-rose-700">
              Red alerts
            </p>
            <p className="mt-3 text-3xl font-semibold text-rose-800">
              {doctorStats.red}
            </p>
            <p className="mt-2 text-sm text-rose-700">Immediate follow-up</p>
          </div>
        </section>

        <section className="rounded-2xl border border-amber-100 bg-white p-6 shadow-card">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-xl font-semibold">Senior overview</h2>
              <p className="text-sm text-stone-600">
                Latest check-ins and context for follow-up.
              </p>
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
              <div className="rounded-2xl border border-rose-100 bg-red-50/60 p-5 text-sm text-rose-700">
                {doctorError}
              </div>
            ) : formattedSeniors.length === 0 ? (
              <div className="rounded-2xl border border-stone-200 bg-stone-50/60 p-5 text-sm text-stone-600">
                No seniors found yet.
              </div>
            ) : null}
            {formattedSeniors.map((senior) => {
              const triageLabel = normalizeTriageLabel(senior.triage);
              return (
                <div
                  key={senior.id}
                  className="flex flex-col gap-4 rounded-2xl border border-stone-200 bg-stone-50/60 p-5 md:flex-row md:items-center md:justify-between"
                >
                  <div>
                    <p className="text-lg font-semibold text-stone-900">
                      {senior.displayName}
                    </p>
                    <p className="text-sm text-stone-600">
                      Last check-in {senior.lastCheckinAt}
                    </p>
                    <p className="mt-2 text-sm text-stone-600">{senior.email}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className={`rounded-full px-4 py-2 text-xs font-semibold ${
                        statusColor[triageLabel] || statusColor.neutral
                      }`}
                    >
                      {triageLabel}
                    </span>
                    <button
                      onClick={() => openReport(senior)}
                      className="rounded-full border border-stone-900 px-4 py-2 text-xs font-semibold text-stone-900 hover:bg-stone-900 hover:text-white"
                    >
                      View report
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </main>

      {reportOpen ? (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/40 px-4 py-10">
          <div className="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-[28px] border border-amber-100 bg-white p-8 shadow-hero">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                  Senior report
                </p>
                <h3 className="mt-2 text-2xl font-semibold text-stone-900">
                  {selectedSenior?.displayName || "Senior"}
                </h3>
                <p className="text-sm text-stone-600">
                  {selectedSenior?.email || ""}
                </p>
              </div>
              <button
                onClick={closeReport}
                className="rounded-full border border-stone-300 px-4 py-2 text-sm font-semibold text-stone-700 hover:border-stone-500"
              >
                Close
              </button>
            </div>

            {reportLoading ? (
              <div className="mt-6 rounded-2xl border border-stone-200 bg-stone-50/60 p-5 text-sm text-stone-600">
                Generating report...
              </div>
            ) : reportError ? (
              <div className="mt-6 rounded-2xl border border-red-100 bg-red-50/60 p-5 text-sm text-rose-700">
                {reportError}
              </div>
            ) : reportData ? (
              <div className="mt-6 grid gap-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-amber-100 bg-amber-50/70 p-5">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-700">
                      Overview
                    </p>
                    <p className="mt-3 text-2xl font-semibold text-stone-900">
                      {reportData.stats.total} check-ins
                    </p>
                    <p className="mt-2 text-sm text-stone-600">
                      Last check-in: {reportData.stats.lastCheckinAt || "None"}
                    </p>
                    <p className="text-sm text-stone-600">
                      Days since last: {reportData.stats.daysSinceLast ?? "—"}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-stone-200 bg-stone-50/60 p-5">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                      Triage mix
                    </p>
                    <div className="mt-3 flex flex-wrap gap-3 text-sm text-stone-700">
                      <span className="rounded-full bg-emerald-100 px-3 py-1">
                        Green {reportData.stats.triageCounts.green}
                      </span>
                      <span className="rounded-full bg-amber-100 px-3 py-1">
                        Yellow {reportData.stats.triageCounts.yellow}
                      </span>
                      <span className="rounded-full bg-red-100 px-3 py-1">
                        Red {reportData.stats.triageCounts.red}
                      </span>
                    </div>
                    <div className="mt-4 overflow-hidden rounded-2xl border border-stone-200 bg-white">
                      <table className="w-full text-left text-sm text-stone-700">
                        <thead className="bg-stone-100 text-xs uppercase tracking-[0.2em] text-stone-500">
                          <tr>
                            <th className="px-3 py-2">Signal</th>
                            <th className="px-3 py-2">Count</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr className="border-t border-stone-200">
                            <td className="px-3 py-2">Dizziness</td>
                            <td className="px-3 py-2">{reportData.signals.dizziness}</td>
                          </tr>
                          <tr className="border-t border-stone-200">
                            <td className="px-3 py-2">Chest pain</td>
                            <td className="px-3 py-2">{reportData.signals.chest_pain}</td>
                          </tr>
                          <tr className="border-t border-stone-200">
                            <td className="px-3 py-2">Breathing issues</td>
                            <td className="px-3 py-2">{reportData.signals.breathing}</td>
                          </tr>
                          <tr className="border-t border-stone-200">
                            <td className="px-3 py-2">Medications missed</td>
                            <td className="px-3 py-2">{reportData.signals.medication_missed}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-stone-200 bg-white p-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                    AI summary
                  </p>
                  {reportData.summaryData ? (
                    <div className="mt-4 grid gap-4 text-sm text-stone-700">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                          Summary
                        </p>
                        <p className="mt-2">{reportData.summaryData.summary}</p>
                      </div>
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                          Symptoms
                        </p>
                        {reportData.summaryData.symptoms?.length ? (
                          <ul className="mt-2 list-disc pl-5">
                            {reportData.summaryData.symptoms.map((item, index) => (
                              <li key={`symptom-${index}`}>{item}</li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-2 text-stone-500">None noted.</p>
                        )}
                      </div>
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                          Risks
                        </p>
                        {reportData.summaryData.risks?.length ? (
                          <ul className="mt-2 list-disc pl-5">
                            {reportData.summaryData.risks.map((item, index) => (
                              <li key={`risk-${index}`}>{item}</li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-2 text-stone-500">None noted.</p>
                        )}
                      </div>
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                          Follow-up
                        </p>
                        {reportData.summaryData.follow_up?.length ? (
                          <ul className="mt-2 list-disc pl-5">
                            {reportData.summaryData.follow_up.map((item, index) => (
                              <li key={`followup-${index}`}>{item}</li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-2 text-stone-500">None noted.</p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="mt-4 text-sm text-stone-600">
                      AI summary unavailable right now. Please try again later.
                    </p>
                  )}
                  <p className="mt-3 text-xs text-stone-500">
                    Generated {reportData.generatedAt}. Informational only; not a diagnosis.
                  </p>
                </div>

                <div className="rounded-2xl border border-stone-200 bg-white p-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">
                    Check-in history
                  </p>
                  {reportData.checkins.length === 0 ? (
                    <p className="mt-4 text-sm text-stone-600">
                      No check-ins recorded yet.
                    </p>
                  ) : (
                    <div className="mt-4 grid gap-3">
                      {reportData.checkins.slice(0, 10).map((checkin) => {
                        const triageLabel = normalizeTriageLabel(
                          checkin.triage_status,
                        );
                        return (
                          <div
                            key={checkin.checkin_id}
                            className="rounded-2xl border border-stone-200 bg-stone-50/60 p-4"
                          >
                            <div className="flex flex-wrap items-center gap-3 text-sm text-stone-700">
                              <span className="font-semibold">
                                {formatDateTime(checkin.completed_at)}
                              </span>
                              <span
                                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                                  statusColor[triageLabel] || statusColor.neutral
                                }`}
                              >
                                {triageLabel}
                              </span>
                            </div>
                            {checkin.triage_reasons?.length ? (
                              <p className="mt-2 text-sm text-stone-600">
                                Outcome: {checkin.triage_reasons.join("; ")}
                              </p>
                            ) : (
                              <p className="mt-2 text-sm text-stone-500">
                                Outcome: No triage notes recorded.
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
