/**
 * Custom hook for state with validator.
 * Provides state with validation before setting.
 */

import { useState, useCallback } from 'react';

/**
 * Validator function type.
 */
export type Validator<T> = (value: T) => boolean | string;

/**
 * Options for useStateWithValidator hook.
 */
export interface UseStateWithValidatorOptions {
  onValidationFail?: (error: string) => void;
}

/**
 * Custom hook for state with validator.
 * Provides state with validation before setting.
 *
 * @param initialValue - Initial state value
 * @param validator - Validator function
 * @param options - Validator options
 * @returns State with validation
 */
export function useStateWithValidator<T>(
  initialValue: T,
  validator: Validator<T>,
  options: UseStateWithValidatorOptions = {}
) {
  const { onValidationFail } = options;
  const [state, setState] = useState<T>(initialValue);
  const [isValid, setIsValid] = useState(() => {
    const result = validator(initialValue);
    return result === true;
  });
  const [error, setError] = useState<string | null>(null);

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      const newValue = typeof value === 'function' ? (value as (prev: T) => T)(state) : value;
      const result = validator(newValue);

      if (result === true) {
        setState(newValue);
        setIsValid(true);
        setError(null);
      } else {
        const errorMessage = typeof result === 'string' ? result : 'Validation failed';
        setIsValid(false);
        setError(errorMessage);
        onValidationFail?.(errorMessage);
      }
    },
    [state, validator, onValidationFail]
  );

  return {
    state,
    setState: setValue,
    isValid,
    error,
  };
}

