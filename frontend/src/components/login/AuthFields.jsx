const LABEL_CLASS =
  "text-sm font-semibold uppercase tracking-[0.08em] text-stone-700";
const INPUT_CLASS =
  "mt-1 w-full rounded-xl border border-[#c9c2b8] bg-[#f6f6f6] px-4 py-3 text-base text-stone-800 outline-none transition focus:border-stone-700 focus:ring-2 focus:ring-stone-200 disabled:cursor-not-allowed disabled:opacity-70";

export default function AuthFields({
  authMode,
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
}) {
  return (
    <div className="mt-3.5 grid gap-3">
      {authMode === "register" ? (
        <div className="grid grid-cols-2 gap-2.5">
          <label className="block">
            <span className={LABEL_CLASS}>First name</span>
            <input
              className={INPUT_CLASS}
              type="text"
              value={authFirstName}
              onChange={(event) => setAuthFirstName(event.target.value)}
              placeholder="John"
              disabled={isLoading}
            />
          </label>
          <label className="block">
            <span className={LABEL_CLASS}>Last name</span>
            <input
              className={INPUT_CLASS}
              type="text"
              value={authLastName}
              onChange={(event) => setAuthLastName(event.target.value)}
              placeholder="Doe"
              disabled={isLoading}
            />
          </label>
        </div>
      ) : null}

      <label className="block">
        <span className={LABEL_CLASS}>Email address</span>
        <input
          className={INPUT_CLASS}
          autoComplete="email"
          inputMode="email"
          value={authEmail}
          onChange={(event) => setAuthEmail(event.target.value)}
          placeholder="bob@gmail.com"
          disabled={isLoading}
        />
      </label>

      <label className="block">
        <span className={LABEL_CLASS}>Password</span>
        <input
          className={INPUT_CLASS}
          type="password"
          autoComplete={authMode === "register" ? "new-password" : "current-password"}
          value={authPassword}
          onChange={(event) => setAuthPassword(event.target.value)}
          placeholder="********"
          disabled={isLoading}
        />
      </label>

      {authError ? <p className="text-sm font-semibold text-rose-700">{authError}</p> : null}
    </div>
  );
}
