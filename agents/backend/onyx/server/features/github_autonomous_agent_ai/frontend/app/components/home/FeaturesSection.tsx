'use client';

import { useMemo } from 'react';

interface Feature {
  id: string;
  title: string;
  description: string;
}

const FEATURES: Feature[] = [
  {
    id: 'ai-ide-core',
    title: 'An AI IDE Core',
    description: "bulk' Editor view offer tab autocompletion, natural language code command, and a configurable, and context-aware configurable agent.",
  },
  {
    id: 'higher-level-abstraction',
    title: 'Higher-level Abstraction',
    description: 'A more intuitive task-based approach to monitoring agent activity, presenting you with essential artifacts and verification results to build trust.',
  },
  {
    id: 'cross-surface-agent',
    title: 'Cross-surface Agent',
    description: 'Synchronized agentic control across your editor, terminal, and browser for powerful development workflows.',
  },
  {
    id: 'user-feedback',
    title: 'User Feedback',
    description: "Intuitively integrate feedback across surfaces and artifacts to guide and refine the agent's work.",
  },
  {
    id: 'agent-first-experience',
    title: 'An Agent-First Experience',
    description: 'Manage multiple agents at the same time, across any workspace, from one central mission control view.',
  },
];

export function FeaturesSection() {
  const features = useMemo(() => FEATURES, []);

  return (
    <section 
      className="py-16 md:py-20 lg:py-24 border-b border-gray-200 relative z-10"
      aria-labelledby="features-heading"
    >
      <div className="max-w-4xl mx-auto px-4 md:px-6 lg:px-8">
        <h2 id="features-heading" className="sr-only">
          Key Features
        </h2>
        <ul className="space-y-8 md:space-y-10 mb-10 md:mb-12" role="list">
          {features.map((feature) => (
            <li 
              key={feature.id} 
              className="text-black text-lg md:text-xl leading-[1.5] font-normal tracking-[-0.01em]"
            >
              <strong className="font-semibold">{feature.title}</strong> - {feature.description}
            </li>
          ))}
        </ul>
        <a 
          href="#product" 
          className="text-black hover:opacity-70 underline transition-opacity duration-200 ease-in-out inline-block text-base leading-normal font-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
          aria-label="Explore product features"
        >
          Explore Product
        </a>
      </div>
    </section>
  );
}

