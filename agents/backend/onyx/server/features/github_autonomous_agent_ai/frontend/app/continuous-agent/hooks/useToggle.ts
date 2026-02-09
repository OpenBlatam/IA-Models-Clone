/**
 * Custom hook for boolean toggle state
 * 
 * Provides convenient toggle functionality for boolean state
 */

import { useState, useCallback } from "react";

/**
 * Return type for useToggle hook
 */
export interface UseToggleReturn {
  /** Current value */
  readonly value: boolean;
  /** Toggle the value */
  readonly toggle: () => void;
  /** Set value to true */
  readonly setTrue: () => void;
  /** Set value to false */
  readonly setFalse: () => void;
  /** Set value directly */
  readonly setValue: (value: boolean) => void;
}

/**
 * Custom hook for boolean toggle state
 * 
 * @param initialValue - Initial boolean value
 * @returns Toggle state and controls
 * 
 * @example
 * ```typescript
 * const { value: isOpen, toggle, setTrue, setFalse } = useToggle(false);
 * 
 * <button onClick={toggle}>Toggle</button>
 * <button onClick={setTrue}>Open</button>
 * <button onClick={setFalse}>Close</button>
 * ```
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




