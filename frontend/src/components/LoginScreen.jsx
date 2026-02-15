import AuthPanel from "../Auth.jsx";
import { API_BASE } from "../lib/api.js";

export default function LoginScreen({ auth }) {
  const {
    authStatus,
    authToken,
    authUser,
    refreshMe,
    logout,
    submitAuth,
    authMode,
    setAuthMode,
    authFirstName,
    setAuthFirstName,
    authLastName,
    setAuthLastName,
    authEmail,
    setAuthEmail,
    authPassword,
    setAuthPassword,
    authError,
  } = auth;

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
          apiBase={API_BASE}
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
