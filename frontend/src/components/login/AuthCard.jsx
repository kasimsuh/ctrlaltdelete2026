import ModeToggle from "./ModeToggle.jsx";
import AuthFields from "./AuthFields.jsx";
import PrimarySubmit from "./PrimarySubmit.jsx";
import AuthedPanel from "./AuthedPanel.jsx";

export default function AuthCard({
  isAuthed,
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
  isLoading,
  ctaLabel,
  authUser,
  authToken,
  refreshMe,
  logout,
}) {
  return (
    <section className="w-full rounded-[22px] border border-[#e8e2d8] bg-[#f7f7f7] p-3 shadow-[0_12px_24px_rgba(44,39,34,0.08)] sm:p-4">
      {isAuthed ? (
        <AuthedPanel
          authUser={authUser}
          authToken={authToken}
          refreshMe={refreshMe}
          logout={logout}
        />
      ) : (
        <form onSubmit={submitAuth}>
          <ModeToggle authMode={authMode} isLoading={isLoading} setAuthMode={setAuthMode} />
          <AuthFields
            authMode={authMode}
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
          />
          <PrimarySubmit ctaLabel={ctaLabel} isLoading={isLoading} />
        </form>
      )}
    </section>
  );
}
