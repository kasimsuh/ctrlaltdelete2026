import { Loader2 } from "lucide-react";

export default function PrimarySubmit({ ctaLabel, isLoading }) {
  return (
    <button
      type="submit"
      className="mt-6 inline-flex h-[52px] w-full items-center justify-center rounded-full bg-gradient-to-b from-[#e46535] to-[#d8542a] px-6 text-lg font-semibold text-white shadow-[0_10px_18px_rgba(222,91,47,0.28)] transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-70 sm:h-[56px] sm:text-xl"
      disabled={isLoading}
    >
      {isLoading ? (
        <span className="inline-flex items-center justify-center">
          <Loader2 aria-hidden="true" className="h-6 w-6 animate-spin" />
          <span className="sr-only">Loading</span>
        </span>
      ) : (
        ctaLabel
      )}
    </button>
  );
}
