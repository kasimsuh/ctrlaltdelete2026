export default function TrustBadges() {
  return (
    <div className="mt-0 flex flex-wrap items-center justify-center gap-x-6 gap-y-1.5 text-sm text-stone-600">
      <div className="inline-flex items-center gap-2">
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          className="h-4 w-4 text-[#d85a2f]"
          fill="none"
        >
          <path
            d="M12 3L5 6v6c0 5 3.4 8.7 7 9.9 3.6-1.2 7-4.9 7-9.9V6l-7-3z"
            stroke="currentColor"
            strokeWidth="1.9"
            strokeLinejoin="round"
          />
        </svg>
        <span className="font-medium">Private &amp; secure</span>
      </div>
      <div className="inline-flex items-center gap-2">
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          className="h-4 w-4 text-[#0f9f4f]"
          fill="currentColor"
        >
          <path d="M12 20.2l-1-.9C6.4 15 3.3 12.2 3.3 8.7A4.6 4.6 0 017.9 4c1.6 0 3.2.8 4.1 2.1A5.2 5.2 0 0116.1 4a4.6 4.6 0 014.6 4.7c0 3.5-3.1 6.3-7.7 10.6l-1 .9z" />
        </svg>
        <span className="font-medium">Senior friendly</span>
      </div>
    </div>
  );
}
