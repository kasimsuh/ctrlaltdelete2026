export default function AuthedPanel({ authUser, authToken, refreshMe, logout }) {
  return (
    <div className="rounded-[16px] border border-[#e4ddd2] bg-white p-3.5 sm:p-4">
      <p className="text-center text-sm text-stone-600 sm:text-base">
        Signed in as <span className="font-semibold text-stone-900">{authUser.email}</span>
      </p>
      <div className="mt-3 grid grid-cols-2 gap-2.5">
        <button
          type="button"
          className="rounded-full border border-stone-300 bg-white px-4 py-2 text-sm font-semibold text-stone-800 transition hover:bg-stone-50"
          onClick={() => refreshMe(authToken)}
        >
          Refresh
        </button>
        <button
          type="button"
          className="rounded-full bg-stone-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-stone-800"
          onClick={logout}
        >
          Sign out
        </button>
      </div>
    </div>
  );
}
