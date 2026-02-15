import useDoctorDashboard from "../hooks/useDoctorDashboard.js";
import { statusColor } from "../lib/screening.js";

export default function DoctorDashboard({ authUser, authToken, logout }) {
  const { doctorStats, formattedSeniors, doctorLoading, doctorError } =
    useDoctorDashboard({ authToken, isDoctor: true });

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
          <div className="rounded-2xl border border-rose-100 bg-rose-50/60 p-6 shadow-card">
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
