'use client';

import React from 'react';
import { Check } from 'lucide-react';
import { clsx } from 'clsx';

interface Step {
  id: string;
  label: string;
  description?: string;
}

interface StepperProps {
  steps: Step[];
  currentStep: number;
  className?: string;
}

export const Stepper: React.FC<StepperProps> = ({
  steps,
  currentStep,
  className,
}) => {
  return (
    <div className={clsx('w-full', className)}>
      <div className="flex items-center justify-between">
        {steps.map((step, index) => {
          const isCompleted = index < currentStep;
          const isCurrent = index === currentStep;
          const isPending = index > currentStep;

          return (
            <React.Fragment key={step.id}>
              <div className="flex flex-col items-center flex-1">
                <div
                  className={clsx(
                    'flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors',
                    isCompleted &&
                      'bg-primary-600 border-primary-600 text-white',
                    isCurrent &&
                      'bg-primary-50 border-primary-600 text-primary-600 dark:bg-primary-900/20 dark:text-primary-400',
                    isPending &&
                      'bg-white border-gray-300 text-gray-400 dark:bg-gray-800 dark:border-gray-700'
                  )}
                >
                  {isCompleted ? (
                    <Check className="h-5 w-5" />
                  ) : (
                    <span className="font-semibold">{index + 1}</span>
                  )}
                </div>
                <div className="mt-2 text-center">
                  <p
                    className={clsx(
                      'text-sm font-medium',
                      isCompleted || isCurrent
                        ? 'text-gray-900 dark:text-white'
                        : 'text-gray-500 dark:text-gray-400'
                    )}
                  >
                    {step.label}
                  </p>
                  {step.description && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {step.description}
                    </p>
                  )}
                </div>
              </div>
              {index < steps.length - 1 && (
                <div
                  className={clsx(
                    'flex-1 h-0.5 mx-4 transition-colors',
                    isCompleted
                      ? 'bg-primary-600'
                      : 'bg-gray-300 dark:bg-gray-700'
                  )}
                />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};


