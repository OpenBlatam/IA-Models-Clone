'use client';

import { ReactNode } from 'react';
import { FiCheck } from 'react-icons/fi';
import { cn } from '@/utils/classNames';

interface Step {
  id: string;
  label: string;
  description?: string;
  icon?: ReactNode;
}

interface StepperProps {
  steps: Step[];
  currentStep: number;
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

export function Stepper({
  steps,
  currentStep,
  orientation = 'horizontal',
  className,
}: StepperProps) {
  if (orientation === 'vertical') {
    return (
      <div className={cn('space-y-4', className)}>
        {steps.map((step, index) => {
          const isCompleted = index < currentStep;
          const isCurrent = index === currentStep;
          const isPending = index > currentStep;

          return (
            <div key={step.id} className="flex gap-4">
              <div className="flex flex-col items-center">
                <div
                  className={cn(
                    'w-10 h-10 rounded-full flex items-center justify-center border-2 font-semibold',
                    isCompleted
                      ? 'bg-primary-600 border-primary-600 text-white'
                      : isCurrent
                      ? 'border-primary-600 text-primary-600 bg-white dark:bg-gray-800'
                      : 'border-gray-300 dark:border-gray-600 text-gray-400 bg-white dark:bg-gray-800'
                  )}
                >
                  {isCompleted ? (
                    <FiCheck size={20} />
                  ) : step.icon ? (
                    step.icon
                  ) : (
                    index + 1
                  )}
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={cn(
                      'w-0.5 h-full min-h-[40px] mt-2',
                      isCompleted
                        ? 'bg-primary-600'
                        : 'bg-gray-300 dark:bg-gray-600'
                    )}
                  />
                )}
              </div>
              <div className="flex-1 pb-8">
                <h3
                  className={cn(
                    'font-semibold',
                    isCurrent
                      ? 'text-primary-600 dark:text-primary-400'
                      : isCompleted
                      ? 'text-gray-900 dark:text-white'
                      : 'text-gray-400 dark:text-gray-500'
                  )}
                >
                  {step.label}
                </h3>
                {step.description && (
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
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
    <div className={cn('w-full', className)}>
      <div className="flex items-center justify-between">
        {steps.map((step, index) => {
          const isCompleted = index < currentStep;
          const isCurrent = index === currentStep;
          const isPending = index > currentStep;

          return (
            <div key={step.id} className="flex items-center flex-1">
              <div className="flex flex-col items-center flex-1">
                <div
                  className={cn(
                    'w-10 h-10 rounded-full flex items-center justify-center border-2 font-semibold',
                    isCompleted
                      ? 'bg-primary-600 border-primary-600 text-white'
                      : isCurrent
                      ? 'border-primary-600 text-primary-600 bg-white dark:bg-gray-800'
                      : 'border-gray-300 dark:border-gray-600 text-gray-400 bg-white dark:bg-gray-800'
                  )}
                >
                  {isCompleted ? (
                    <FiCheck size={20} />
                  ) : step.icon ? (
                    step.icon
                  ) : (
                    index + 1
                  )}
                </div>
                <div className="mt-2 text-center">
                  <p
                    className={cn(
                      'text-sm font-medium',
                      isCurrent
                        ? 'text-primary-600 dark:text-primary-400'
                        : isCompleted
                        ? 'text-gray-900 dark:text-white'
                        : 'text-gray-400 dark:text-gray-500'
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
                  className={cn(
                    'flex-1 h-0.5 mx-4',
                    isCompleted
                      ? 'bg-primary-600'
                      : 'bg-gray-300 dark:bg-gray-600'
                  )}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

