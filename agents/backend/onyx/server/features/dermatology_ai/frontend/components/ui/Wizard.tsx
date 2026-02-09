'use client';

import React, { useState } from 'react';
import { Check } from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from './Button';

interface WizardStep {
  id: string;
  title: string;
  description?: string;
  content: React.ReactNode;
}

interface WizardProps {
  steps: WizardStep[];
  onComplete?: () => void;
  className?: string;
}

export const Wizard: React.FC<WizardProps> = ({
  steps,
  onComplete,
  className,
}) => {
  const [currentStep, setCurrentStep] = useState(0);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete?.();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleStepClick = (index: number) => {
    setCurrentStep(index);
  };

  return (
    <div className={clsx('w-full', className)}>
      {/* Steps Indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, index) => {
            const isCompleted = index < currentStep;
            const isCurrent = index === currentStep;

            return (
              <React.Fragment key={step.id}>
                <div className="flex flex-col items-center flex-1">
                  <button
                    onClick={() => handleStepClick(index)}
                    className={clsx(
                      'w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-all',
                      isCompleted
                        ? 'bg-primary-600 text-white'
                        : isCurrent
                        ? 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400 border-2 border-primary-600'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                    )}
                  >
                    {isCompleted ? (
                      <Check className="h-5 w-5" />
                    ) : (
                      index + 1
                    )}
                  </button>
                  <div className="mt-2 text-center">
                    <p
                      className={clsx(
                        'text-sm font-medium',
                        isCurrent
                          ? 'text-primary-600 dark:text-primary-400'
                          : 'text-gray-600 dark:text-gray-400'
                      )}
                    >
                      {step.title}
                    </p>
                    {step.description && (
                      <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                        {step.description}
                      </p>
                    )}
                  </div>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={clsx(
                      'flex-1 h-0.5 mx-2',
                      isCompleted
                        ? 'bg-primary-600'
                        : 'bg-gray-200 dark:bg-gray-700'
                    )}
                  />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Step Content */}
      <div className="mb-8">{steps[currentStep].content}</div>

      {/* Navigation */}
      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={handlePrevious}
          disabled={currentStep === 0}
        >
          Anterior
        </Button>
        <Button onClick={handleNext}>
          {currentStep === steps.length - 1 ? 'Completar' : 'Siguiente'}
        </Button>
      </div>
    </div>
  );
};


