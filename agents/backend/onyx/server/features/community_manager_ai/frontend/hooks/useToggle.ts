/**
 * useToggle Hook
 * Composable hook for boolean state management
 */

import { useState, useCallback } from 'react';

/**
 * Hook to manage boolean state with toggle functionality
 * @param initialValue - Initial boolean value (default: false)
 * @returns Tuple with [value, toggle, setTrue, setFalse, setValue]
 */
export const useToggle = (initialValue: boolean = false) => {
  const [value, setValue] = useState(initialValue);

  const toggle = useCallback(() => setValue((prev) => !prev), []);
  const setTrue = useCallback(() => setValue(true), []);
  const setFalse = useCallback(() => setValue(false), []);
  const setValueCallback = useCallback((newValue: boolean | ((prev: boolean) => boolean)) => {
    setValue(typeof newValue === 'function' ? newValue : () => newValue);
  }, []);

  return [value, toggle, setTrue, setFalse, setValueCallback] as const;
};


