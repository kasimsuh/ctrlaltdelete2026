const PAGE_SHELL_CLASS = "min-h-screen bg-[#f3f0ea] px-3 py-2 text-[#1d1b19] sm:px-4 sm:py-3";

function SuccessCheckIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-9 w-9" fill="none" stroke="currentColor" strokeWidth="2.5">
      <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function CompletionScreen() {
  return (
    <div className={PAGE_SHELL_CLASS}>
      <main className="mx-auto flex min-h-[calc(100svh-1rem)] w-full max-w-3xl items-center justify-center sm:min-h-[calc(100svh-1.5rem)]">
        <div className="w-full rounded-[28px] border border-[#e8e2d8] bg-[#f7f7f7] p-8 text-center shadow-[0_18px_36px_rgba(44,39,34,0.1)] sm:p-10">
          <div className="mx-auto grid h-16 w-16 place-items-center rounded-full bg-emerald-100 text-emerald-600">
            <SuccessCheckIcon />
          </div>
          <h1 className="mt-6 text-2xl font-semibold text-stone-900 sm:text-3xl">
            Thank you for your input.
          </h1>
          <p className="mt-2 text-base text-stone-600">Have a wonderful day!</p>
        </div>
      </main>
    </div>
  );
}
