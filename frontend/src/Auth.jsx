export default function AuthPanel({
  authStatus,
  authToken,
  authUser,
  apiBase,
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
  authError
}) {
  const isLoading = authStatus === "loading";
  const isAuthed = Boolean(authToken && authUser?.email);

  return (
    <section className="rounded-3xl border border-amber-200/80 bg-white/70 p-6 shadow-[0_18px_40px_rgba(31,27,22,0.10)] backdrop-blur">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-500">Account</p>
          <h2 className="mt-2 text-2xl font-semibold text-stone-900">User authentication</h2>
        </div>

        {isAuthed ? (
          <div className="flex flex-col gap-2 rounded-2xl border border-emerald-200 bg-emerald-50/60 p-4">
            <div className="text-sm text-stone-700">
              Signed in as <span className="font-semibold">{authUser.email}</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                className="rounded-full bg-stone-900 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-800"
                onClick={() => refreshMe(authToken)}
              >
                Refresh
              </button>
              <button
                type="button"
                className="rounded-full border border-stone-300 bg-white px-4 py-2 text-sm font-semibold text-stone-800 hover:bg-stone-50"
                onClick={logout}
              >
                Log out
              </button>
            </div>
          </div>
        ) : (
          <form className="w-full max-w-md" onSubmit={submitAuth}>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className={`rounded-full px-4 py-2 text-sm font-semibold ${
                  authMode === "login"
                    ? "bg-stone-900 text-white"
                    : "border border-stone-300 bg-white text-stone-800 hover:bg-stone-50"
                }`}
                onClick={() => setAuthMode("login")}
                disabled={isLoading}
              >
                Log in
              </button>
              <button
                type="button"
                className={`rounded-full px-4 py-2 text-sm font-semibold ${
                  authMode === "register"
                    ? "bg-stone-900 text-white"
                    : "border border-stone-300 bg-white text-stone-800 hover:bg-stone-50"
                }`}
                onClick={() => setAuthMode("register")}
                disabled={isLoading}
              >
                Register
              </button>
            </div>

            <div className="mt-4 grid gap-3">
              {authMode === "register" && (
                <>
                  <label className="grid gap-1">
                    <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">First Name</span>
                    <input
                      className="rounded-2xl border border-stone-300 bg-white px-4 py-3 text-sm outline-none focus:border-stone-900"
                      type="text"
                      value={authFirstName}
                      onChange={(e) => setAuthFirstName(e.target.value)}
                      placeholder="John"
                      disabled={isLoading}
                    />
                  </label>

                  <label className="grid gap-1">
                    <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">Last Name</span>
                    <input
                      className="rounded-2xl border border-stone-300 bg-white px-4 py-3 text-sm outline-none focus:border-stone-900"
                      type="text"
                      value={authLastName}
                      onChange={(e) => setAuthLastName(e.target.value)}
                      placeholder="Doe"
                      disabled={isLoading}
                    />
                  </label>
                </>
              )}

              <label className="grid gap-1">
                <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">Email</span>
                <input
                  className="rounded-2xl border border-stone-300 bg-white px-4 py-3 text-sm outline-none focus:border-stone-900"
                  autoComplete="email"
                  inputMode="email"
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  placeholder="you@example.com"
                  disabled={isLoading}
                />
              </label>

              <label className="grid gap-1">
                <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">Password</span>
                <input
                  className="rounded-2xl border border-stone-300 bg-white px-4 py-3 text-sm outline-none focus:border-stone-900"
                  type="password"
                  autoComplete={authMode === "register" ? "new-password" : "current-password"}
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  placeholder="••••••••"
                  disabled={isLoading}
                />
              </label>

              {authError ? <p className="text-sm font-medium text-rose-700">{authError}</p> : null}

              <button
                type="submit"
                className="rounded-full bg-[#de5b2f] px-5 py-3 text-sm font-semibold text-white shadow-[0_12px_20px_rgba(222,91,47,0.25)] hover:translate-y-[-1px] hover:bg-[#c94f27] disabled:cursor-not-allowed disabled:opacity-70"
                disabled={isLoading}
              >
                {isLoading ? "Working..." : authMode === "register" ? "Create account" : "Sign in"}
              </button>
            </div>
          </form>
        )}
      </div>
    </section>
  );
}
