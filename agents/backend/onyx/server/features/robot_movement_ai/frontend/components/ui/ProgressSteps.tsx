'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { Check } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface Step {
  id: string;
  title: string;
  description?: string;
  icon?: ReactNode;
}

interface ProgressStepsProps {
  steps: Step[];
  currentStep: number;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
}

export default function ProgressSteps({
  steps,
  currentStep,
  className,
  orientation = 'horizontal',
}: ProgressStepsProps) {
  if (orientation === 'vertical') {
    return (
      <div className={cn('flex flex-col gap-4', className)}>
        {steps.map((step, index) => (
          <StepVertical
            key={step.id}
            step={step}
            index={index}
            isActive={index === currentStep}
            isCompleted={index < currentStep}
          />
        ))}
      </div>
    );
  }

  return (
    <div className={cn('relative', className)}>
      {/* Progress Line */}
      <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gray-200 -translate-y-1/2" />
      <motion.div
        className="absolute top-1/2 left-0 h-0.5 bg-tesla-blue -translate-y-1/2"
        initial={{ width: 0 }}
        animate={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
        transition={{ duration: 0.3 }}
      />

      {/* Steps */}
      <div className="relative flex justify-between">
        {steps.map((step, index) => (
          <StepHorizontal
            key={step.id}
            step={step}
            index={index}
            isActive={index === currentStep}
            isCompleted={index < currentStep}
          />
        ))}
      </div>
    </div>
  );
}

function StepHorizontal({
  step,
  index,
  isActive,
  isCompleted,
}: {
  step: Step;
  index: number;
  isActive: boolean;
  isCompleted: boolean;
}) {
  return (
    <div className="flex flex-col items-center relative z-10">
      <motion.div
        initial={false}
        animate={{
          scale: isActive ? 1.1 : 1,
          backgroundColor: isCompleted
            ? '#10b981'
            : isActive
            ? '#0062cc'
            : '#e5e7eb',
        }}
        className={cn(
          'w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold transition-colors',
          isActive && 'ring-4 ring-tesla-blue/20'
        )}
      >
        {isCompleted ? (
          <Check className="w-5 h-5" />
        ) : (
          step.icon || <span>{index + 1}</span>
        )}
      </motion.div>
      <div className="mt-2 text-center max-w-[120px]">
        <div
          className={cn(
            'text-sm font-medium',
            isActive ? 'text-tesla-blue' : isCompleted ? 'text-green-600' : 'text-tesla-gray-dark'
          )}
        >
          {step.title}
        </div>
        {step.description && (
          <div className="text-xs text-tesla-gray-dark mt-1">{step.description}</div>
        )}
      </div>
    </div>
  );
}

function StepVertical({
  step,
  index,
  isActive,
  isCompleted,
}: {
  step: Step;
  index: number;
  isActive: boolean;
  isCompleted: boolean;
}) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <motion.div
          initial={false}
          animate={{
            scale: isActive ? 1.1 : 1,
            backgroundColor: isCompleted
              ? '#10b981'
              : isActive
              ? '#0062cc'
              : '#e5e7eb',
          }}
          className={cn(
            'w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold transition-colors flex-shrink-0',
            isActive && 'ring-4 ring-tesla-blue/20'
          )}
        >
          {isCompleted ? (
            <Check className="w-5 h-5" />
          ) : (
            step.icon || <span>{index + 1}</span>
          )}
        </motion.div>
        {index < 2 && (
          <div className="w-0.5 h-full bg-gray-200 mt-2" />
        )}
      </div>
      <div className="flex-1 pb-8">
        <div
          className={cn(
            'text-base font-semibold mb-1',
            isActive ? 'text-tesla-blue' : isCompleted ? 'text-green-600' : 'text-tesla-gray-dark'
          )}
        >
          {step.title}
        </div>
        {step.description && (
          <div className="text-sm text-tesla-gray-dark">{step.description}</div>
        )}
      </div>
    </div>
  );
}



