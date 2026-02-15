import BrandHeader from "./login/BrandHeader.jsx";
import AuthCard from "./login/AuthCard.jsx";
import TrustBadges from "./login/TrustBadges.jsx";

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

  const isLoading = authStatus === "loading";
  const isAuthed = Boolean(authToken && authUser?.email);
  const ctaLabel = isLoading
    ? "Working..."
    : authMode === "register"
      ? "Create account"
      : "Sign in";

  return (
    <div className="h-[100svh] overflow-hidden bg-[#f3f0ea] px-3 py-2 text-[#1d1b19] sm:px-4 sm:py-3">
      <main className="mx-auto grid h-full w-full max-w-5xl place-items-center">
        <section className="flex w-full max-w-[560px] flex-col items-center gap-3 sm:gap-3.5">
          <BrandHeader />
          <AuthCard
            isAuthed={isAuthed}
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
            isLoading={isLoading}
            ctaLabel={ctaLabel}
            authUser={authUser}
            authToken={authToken}
            refreshMe={refreshMe}
            logout={logout}
          />
          {!isAuthed ? <TrustBadges /> : null}
        </section>
      </main>
    </div>
  );
}
