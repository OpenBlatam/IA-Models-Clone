'use client';

import { useState, useCallback } from 'react';

interface UseStepperOptions {
  initialStep?: number;
  totalSteps: number;
  onStepChange?: (step: number) => void;
}

export function useStepper({
  initialStep = 0,
  totalSteps,
  onStepChange,
}: UseStepperOptions) {
  const [currentStep, setCurrentStep] = useState(initialStep);

  const next = useCallback(() => {
    if (currentStep < totalSteps - 1) {
      const newStep = currentStep + 1;
      setCurrentStep(newStep);
      onStepChange?.(newStep);
    }
  }, [currentStep, totalSteps, onStepChange]);

  const previous = useCallback(() => {
    if (currentStep > 0) {
      const newStep = currentStep - 1;
      setCurrentStep(newStep);
      onStepChange?.(newStep);
    }
  }, [currentStep, onStepChange]);

  const goTo = useCallback(
    (step: number) => {
      if (step >= 0 && step < totalSteps) {
        setCurrentStep(step);
        onStepChange?.(step);
      }
    },
    [totalSteps, onStepChange]
  );

  const reset = useCallback(() => {
    setCurrentStep(initialStep);
    onStepChange?.(initialStep);
  }, [initialStep, onStepChange]);

  return {
    currentStep,
    next,
    previous,
    goTo,
    reset,
    isFirstStep: currentStep === 0,
    isLastStep: currentStep === totalSteps - 1,
  };
}

