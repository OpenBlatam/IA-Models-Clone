'use client';

import { Check, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Step {
  id: string;
  label: string;
  description?: string;
}

interface StepperProps {
  steps: Step[];
  currentStep: number;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
}

export const Stepper = ({
  steps,
  currentStep,
  className,
  orientation = 'horizontal',
}: StepperProps) => {
  if (orientation === 'vertical') {
    return (
      <div className={cn('space-y-4', className)}>
        {steps.map((step, index) => {
          const isCompleted = index < currentStep;
          const isCurrent = index === currentStep;
          const isUpcoming = index > currentStep;

          return (
            <div key={step.id} className="flex items-start gap-4">
              <div className="flex flex-col items-center">
                <div
                  className={cn(
                    'flex h-10 w-10 items-center justify-center rounded-full border-2 font-medium transition-colors',
                    isCompleted
                      ? 'border-primary-600 bg-primary-600 text-white dark:bg-primary-500'
                      : isCurrent
                      ? 'border-primary-600 bg-white text-primary-600 dark:bg-gray-800 dark:text-primary-400'
                      : 'border-gray-300 bg-white text-gray-400 dark:border-gray-700 dark:bg-gray-800'
                  )}
                >
                  {isCompleted ? (
                    <Check className="h-5 w-5" />
                  ) : (
                    <span>{index + 1}</span>
                  )}
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={cn(
                      'mt-2 h-full w-0.5',
                      isCompleted
                        ? 'bg-primary-600 dark:bg-primary-500'
                        : 'bg-gray-300 dark:bg-gray-700'
                    )}
                  />
                )}
              </div>
              <div className="flex-1 pb-8">
                <p
                  className={cn(
                    'text-sm font-medium',
                    isCurrent
                      ? 'text-primary-600 dark:text-primary-400'
                      : isCompleted
                      ? 'text-gray-900 dark:text-gray-100'
                      : 'text-gray-500 dark:text-gray-400'
                  )}
                >
                  {step.label}
                </p>
                {step.description && (
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    {step.description}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className={cn('flex items-center', className)}>
      {steps.map((step, index) => {
        const isCompleted = index < currentStep;
        const isCurrent = index === currentStep;
        const isUpcoming = index > currentStep;

        return (
          <div key={step.id} className="flex items-center">
            <div className="flex items-center">
              <div
                className={cn(
                  'flex h-10 w-10 items-center justify-center rounded-full border-2 font-medium transition-colors',
                  isCompleted
                    ? 'border-primary-600 bg-primary-600 text-white dark:bg-primary-500'
                    : isCurrent
                    ? 'border-primary-600 bg-white text-primary-600 dark:bg-gray-800 dark:text-primary-400'
                    : 'border-gray-300 bg-white text-gray-400 dark:border-gray-700 dark:bg-gray-800'
                )}
              >
                {isCompleted ? (
                  <Check className="h-5 w-5" />
                ) : (
                  <span>{index + 1}</span>
                )}
              </div>
              <div className="ml-4 hidden sm:block">
                <p
                  className={cn(
                    'text-sm font-medium',
                    isCurrent
                      ? 'text-primary-600 dark:text-primary-400'
                      : isCompleted
                      ? 'text-gray-900 dark:text-gray-100'
                      : 'text-gray-500 dark:text-gray-400'
                  )}
                >
                  {step.label}
                </p>
                {step.description && (
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    {step.description}
                  </p>
                )}
              </div>
            </div>
            {index < steps.length - 1 && (
              <div className="mx-4 flex items-center">
                <ChevronRight
                  className={cn(
                    'h-5 w-5',
                    isCompleted
                      ? 'text-primary-600 dark:text-primary-400'
                      : 'text-gray-300 dark:text-gray-700'
                  )}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};



