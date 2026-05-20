'use client';

interface DownloadOption {
  id: string;
  label: string;
  ariaLabel: string;
}

const DOWNLOAD_OPTIONS: DownloadOption[] = [
  {
    id: 'x64',
    label: 'Download for x64',
    ariaLabel: 'Download for x64 architecture',
  },
  {
    id: 'arm64',
    label: 'Download for ARM64',
    ariaLabel: 'Download for ARM64 architecture',
  },
];

export function DownloadSection() {
  return (
    <section 
      className="py-16 md:py-20 lg:py-24 border-b border-gray-200 relative z-10"
      aria-labelledby="download-heading"
    >
      <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8">
        <h2 id="download-heading" className="sr-only">
          Download Options
        </h2>
        <div className="flex flex-col sm:flex-row gap-3 md:gap-4 justify-center items-center">
          {DOWNLOAD_OPTIONS.map((option) => (
            <button
              key={option.id}
              className="bg-black text-white px-8 py-4 rounded-lg hover:bg-[#1a1a1a] transition-colors duration-200 ease-in-out font-normal text-base leading-normal min-w-[200px] focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label={option.ariaLabel}
              type="button"
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

