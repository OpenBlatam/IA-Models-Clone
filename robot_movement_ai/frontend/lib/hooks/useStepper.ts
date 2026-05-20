import { useState, useCallback } from 'react';

export interface UseStepperOptions {
  initialStep?: number;
  totalSteps: number;
  loop?: boolean;
}

export interface UseStepperReturn {
  currentStep: number;
  next: () => void;
  previous: () => void;
  goTo: (step: number) => void;
  reset: () => void;
  isFirst: boolean;
  isLast: boolean;
  canGoNext: boolean;
  canGoPrevious: boolean;
}

/**
 * Hook for stepper/wizard navigation
 */
export function useStepper(options: UseStepperOptions): UseStepperReturn {
  const { initialStep = 0, totalSteps, loop = false } = options;
  const [currentStep, setCurrentStep] = useState(initialStep);

  const next = useCallback(() => {
    setCurrentStep((prev) => {
      if (prev < totalSteps - 1) {
        return prev + 1;
      }
      if (loop) {
        return 0;
      }
      return prev;
    });
  }, [totalSteps, loop]);

  const previous = useCallback(() => {
    setCurrentStep((prev) => {
      if (prev > 0) {
        return prev - 1;
      }
      if (loop) {
        return totalSteps - 1;
      }
      return prev;
    });
  }, [totalSteps, loop]);

  const goTo = useCallback(
    (step: number) => {
      if (step >= 0 && step < totalSteps) {
        setCurrentStep(step);
      }
    },
    [totalSteps]
  );

  const reset = useCallback(() => {
    setCurrentStep(initialStep);
  }, [initialStep]);

  const isFirst = currentStep === 0;
  const isLast = currentStep === totalSteps - 1;
  const canGoNext = loop || !isLast;
  const canGoPrevious = loop || !isFirst;

  return {
    currentStep,
    next,
    previous,
    goTo,
    reset,
    isFirst,
    isLast,
    canGoNext,
    canGoPrevious,
  };
}



