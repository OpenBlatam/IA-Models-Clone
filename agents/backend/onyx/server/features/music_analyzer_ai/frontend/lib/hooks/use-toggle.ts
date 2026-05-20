/**
 * Custom hook for toggle state.
 * Provides a simple boolean toggle with optional initial value.
 */

import { useState, useCallback } from 'react';

/**
 * Return type for useToggle hook.
 */
export interface UseToggleReturn {
  value: boolean;
  toggle: () => void;
  setTrue: () => void;
  setFalse: () => void;
  setValue: (value: boolean) => void;
}

/**
 * Custom hook for toggle state.
 * Provides convenient methods for boolean state management.
 *
 * @param initialValue - Initial boolean value (default: false)
 * @returns Toggle state and handlers
 */
export function useToggle(initialValue: boolean = false): UseToggleReturn {
  const [value, setValue] = useState(initialValue);

  const toggle = useCallback(() => {
    setValue((prev) => !prev);
  }, []);

  const setTrue = useCallback(() => {
    setValue(true);
  }, []);

  const setFalse = useCallback(() => {
    setValue(false);
  }, []);

  return {
    value,
    toggle,
    setTrue,
    setFalse,
    setValue,
  };
}

