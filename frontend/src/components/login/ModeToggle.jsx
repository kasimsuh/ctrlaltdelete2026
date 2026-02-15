export default function ModeToggle({ authMode, isLoading, setAuthMode }) {
  return (
    <div className="grid grid-cols-2 gap-1 rounded-full bg-[#ececec] p-1">
      <button
        type="button"
        className={`rounded-full px-3 py-2 text-sm font-semibold transition sm:py-2.5 sm:text-base ${
          authMode === "login"
            ? "bg-[#171513] text-white shadow-[0_8px_16px_rgba(0,0,0,0.24)]"
            : "text-stone-600 hover:text-stone-800"
        }`}
        onClick={() => setAuthMode("login")}
        disabled={isLoading}
      >
        Sign in
      </button>
      <button
        type="button"
        className={`rounded-full px-3 py-2 text-sm font-semibold transition sm:py-2.5 sm:text-base ${
          authMode === "register"
            ? "bg-[#171513] text-white shadow-[0_8px_16px_rgba(0,0,0,0.24)]"
            : "text-stone-600 hover:text-stone-800"
        }`}
        onClick={() => setAuthMode("register")}
        disabled={isLoading}
      >
        Register
      </button>
    </div>
  );
}
