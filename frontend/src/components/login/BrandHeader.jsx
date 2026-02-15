export default function BrandHeader() {
  return (
    <>
      <div className="relative grid h-16 w-16 place-items-center rounded-[19px] bg-gradient-to-b from-[#e46636] to-[#d8542a] shadow-[0_12px_22px_rgba(222,91,47,0.24)] sm:h-20 sm:w-20 sm:rounded-[22px]">
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          className="h-7 w-7 text-white sm:h-8 sm:w-8"
          fill="none"
        >
          <path
            d="M12 3L5 6v6c0 5 3.4 8.7 7 9.9 3.6-1.2 7-4.9 7-9.9V6l-7-3z"
            stroke="currentColor"
            strokeWidth="1.9"
            strokeLinejoin="round"
          />
          <path
            d="M9.6 12.2l1.7 1.7 3.2-3.6"
            stroke="currentColor"
            strokeWidth="1.9"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <div className="absolute -bottom-1 -right-1 grid h-7 w-7 place-items-center rounded-full bg-[#d9f4df] shadow-[0_6px_12px_rgba(15,63,36,0.2)]">
          <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            className="h-3 w-3 text-[#0f9f4f]"
            fill="currentColor"
          >
            <path d="M12 20.2l-1-.9C6.4 15 3.3 12.2 3.3 8.7A4.6 4.6 0 017.9 4c1.6 0 3.2.8 4.1 2.1A5.2 5.2 0 0116.1 4a4.6 4.6 0 014.6 4.7c0 3.5-3.1 6.3-7.7 10.6l-1 .9z" />
          </svg>
        </div>
      </div>

      <p className="-mt-1 text-center text-[clamp(1.7rem,3.8vw,2.6rem)] font-bold tracking-tight text-[#1f1d1b]">
        SeniCare
      </p>

      <h1 className="mt-1 text-center text-[clamp(2rem,5vw,3.4rem)] font-bold leading-[0.98] tracking-tight text-[#1f1d1b]">
        Your daily
        <br />
        wellness companion
      </h1>

      <p className="max-w-[500px] text-center text-[clamp(0.9rem,1.3vw,1.05rem)] leading-[1.25] text-stone-600">
        Quick camera and voice check-in designed
        <br className="hidden sm:block" />
        <span className="sm:hidden"> </span>
        for seniors. Simple, safe, and private.
      </p>
    </>
  );
}
