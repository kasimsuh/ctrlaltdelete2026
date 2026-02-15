export default function BrandHeader() {
  return (
    <>
      <div className="relative h-24 w-24 overflow-hidden rounded-[24px] shadow-[0_12px_22px_rgba(222,91,47,0.24)] sm:h-28 sm:w-28 sm:rounded-[28px]">
        <img
          src="/senicarelogo.png"
          alt="SeniCare logo"
          className="h-full w-full object-cover"
        />
      </div>

      <p className="-mt-1 text-center text-[clamp(2.8rem,7.8vw,5.4rem)] font-bold tracking-tight text-[#1f1d1b]">
        SeniCare
      </p>

      <p className="-mt-2 max-w-[500px] text-center text-[clamp(0.9rem,1.3vw,1.05rem)] leading-[1.25] text-stone-600 sm:-mt-2.5">
        Quick camera and voice check-in designed
        <br className="hidden sm:block" />
        <span className="sm:hidden"> </span>
        for seniors. Simple, safe, and private.
      </p>
    </>
  );
}
