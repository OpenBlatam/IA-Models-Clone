'use client';

import { useMemo } from 'react';

interface UseCase {
  id: string;
  label: string;
  description: string;
  ariaLabel: string;
}

const USE_CASES: UseCase[] = [
  {
    id: 'ux-development',
    label: 'Use Case 1',
    description: 'Streamline UX development by leveraging browser-in-the-loop agents to automate repetitive tasks.',
    ariaLabel: 'View use case 1: UX development case study',
  },
  {
    id: 'production-apps',
    label: 'Use Case 2',
    description: 'Build production-ready applications with confidence with thoroughly designed artifacts and comprehensive verification tests.',
    ariaLabel: 'View use case 2: Production applications case study',
  },
  {
    id: 'operations-orchestration',
    label: 'Use Case 3',
    description: 'Streamline operations and reduce context switching by orchestrating agents across workspaces using the Agent Manager.',
    ariaLabel: 'View use case 3: Operations orchestration case study',
  },
];

export function UseCasesSection() {
  const useCases = useMemo(() => USE_CASES, []);

  return (
    <section 
      className="py-16 md:py-20 lg:py-24 border-b border-gray-200 relative z-10"
      aria-labelledby="use-cases-heading"
    >
      <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8">
        <h2 id="use-cases-heading" className="sr-only">
          Use Cases
        </h2>
        <div className="grid md:grid-cols-3 gap-8 md:gap-10">
          {useCases.map((useCase) => (
            <article key={useCase.id} className="space-y-5">
              <div className="text-gray-500 text-xs uppercase tracking-wide font-medium leading-normal">{useCase.label}</div>
              <p className="text-black text-lg leading-[1.5] font-normal tracking-[-0.01em]">
                {useCase.description}
              </p>
              <a 
                href="#" 
                className="text-black hover:opacity-70 underline transition-opacity duration-200 ease-in-out inline-block text-sm leading-normal font-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
                aria-label={useCase.ariaLabel}
              >
                View case
              </a>
            </article>
          ))}
        </div>
        <div className="flex justify-center gap-3 mt-12" role="group" aria-label="Use cases navigation controls">
          <button 
            className="text-black hover:opacity-70 p-2 rounded-full hover:bg-gray-100 transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
            aria-label="Previous use cases"
            type="button"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <button 
            className="text-black hover:opacity-70 p-2 rounded-full hover:bg-gray-100 transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
            aria-label="Next use cases"
            type="button"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </section>
  );
}

