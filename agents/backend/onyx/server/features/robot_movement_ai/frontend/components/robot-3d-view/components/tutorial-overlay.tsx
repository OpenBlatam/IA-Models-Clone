/**
 * Tutorial Overlay Component
 * @module robot-3d-view/components/tutorial-overlay
 */

'use client';

import { memo, useState, useEffect } from 'react';
import { useConfigPersistence } from '../hooks/use-config-persistence';

/**
 * Tutorial step
 */
interface TutorialStep {
  id: string;
  title: string;
  content: string;
  target?: string; // CSS selector for highlighting
  position?: 'top' | 'bottom' | 'left' | 'right' | 'center';
}

/**
 * Tutorial configuration
 */
interface Tutorial {
  id: string;
  name: string;
  steps: TutorialStep[];
}

/**
 * Default tutorials
 */
const DEFAULT_TUTORIALS: Tutorial[] = [
  {
    id: 'getting-started',
    name: 'Getting Started',
    steps: [
      {
        id: 'welcome',
        title: 'Welcome!',
        content: 'Welcome to the Robot 3D View. This tutorial will guide you through the main features.',
        position: 'center',
      },
      {
        id: 'controls',
        title: 'Controls',
        content: 'Use your mouse to rotate the view (drag), zoom (scroll), and pan (right-click + drag).',
        position: 'bottom',
      },
      {
        id: 'shortcuts',
        title: 'Keyboard Shortcuts',
        content: 'Press H to see all available keyboard shortcuts. Try pressing S to toggle statistics.',
        position: 'bottom',
      },
      {
        id: 'presets',
        title: 'Presets',
        content: 'Use presets to quickly switch between different scene configurations.',
        position: 'top-right',
      },
      {
        id: 'themes',
        title: 'Themes',
        content: 'Customize the appearance with different themes.',
        position: 'top-right',
      },
    ],
  },
];

/**
 * Tutorial Overlay Component
 * 
 * Provides interactive tutorials and onboarding.
 * 
 * @returns Tutorial overlay component
 */
export const TutorialOverlay = memo(() => {
  const { getConfig, setConfig } = useConfigPersistence();
  const [currentTutorial, setCurrentTutorial] = useState<Tutorial | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Check if user has completed tutorials
    const completedTutorials = getConfig('completedTutorials', []) as string[];
    const shouldShow = !completedTutorials.includes('getting-started');

    if (shouldShow) {
      setCurrentTutorial(DEFAULT_TUTORIALS[0]);
      setIsVisible(true);
    }
  }, [getConfig]);

  const handleNext = () => {
    if (!currentTutorial) return;

    if (currentStep < currentTutorial.steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    handleComplete();
  };

  const handleComplete = () => {
    if (!currentTutorial) return;

    const completedTutorials = getConfig('completedTutorials', []) as string[];
    completedTutorials.push(currentTutorial.id);
    setConfig('completedTutorials', completedTutorials);

    setIsVisible(false);
    setCurrentTutorial(null);
    setCurrentStep(0);
  };

  if (!isVisible || !currentTutorial) return null;

  const step = currentTutorial.steps[currentStep];
  const isFirst = currentStep === 0;
  const isLast = currentStep === currentTutorial.steps.length - 1;

  return (
    <div className="absolute inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div
        className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-6 max-w-md w-full mx-4 shadow-xl"
        role="dialog"
        aria-modal="true"
        aria-label="Tutorial"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">{step.title}</h2>
          <button
            onClick={handleSkip}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Skip tutorial"
          >
            ✕
          </button>
        </div>

        <div className="mb-6">
          <p className="text-gray-300">{step.content}</p>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex gap-2">
            <button
              onClick={handlePrevious}
              disabled={isFirst}
              className={`
                px-4 py-2 rounded transition-colors
                ${isFirst ? 'bg-gray-700 text-gray-500 cursor-not-allowed' : 'bg-gray-700 hover:bg-gray-600 text-white'}
              `}
            >
              Previous
            </button>
            <button
              onClick={handleNext}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white transition-colors"
            >
              {isLast ? 'Complete' : 'Next'}
            </button>
          </div>

          <div className="text-sm text-gray-400">
            {currentStep + 1} / {currentTutorial.steps.length}
          </div>
        </div>

        {/* Progress indicator */}
        <div className="mt-4 flex gap-1">
          {currentTutorial.steps.map((_, index) => (
            <div
              key={index}
              className={`
                flex-1 h-1 rounded transition-colors
                ${index <= currentStep ? 'bg-blue-600' : 'bg-gray-700'}
              `}
            />
          ))}
        </div>
      </div>
    </div>
  );
});

TutorialOverlay.displayName = 'TutorialOverlay';



