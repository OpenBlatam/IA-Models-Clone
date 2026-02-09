/**
 * Custom hook for debounced values
 * 
 * Provides debounced value updates for form inputs and search
 */

import { useState, useEffect } from "react";

/**
 * Options for useDebouncedValue hook
 */
export interface UseDebouncedValueOptions {
  /** Delay in milliseconds */
  readonly delay?: number;
  /** Whether to debounce on initial mount */
  readonly immediate?: boolean;
}

/**
 * Return type for useDebouncedValue hook
 */
export interface UseDebouncedValueReturn<T> {
  /** Current value */
  readonly value: T;
  /** Debounced value */
  readonly debouncedValue: T;
  /** Whether value is being debounced */
  readonly isDebouncing: boolean;
  /** Set value */
  readonly setValue: (value: T) => void;
}

const DEFAULT_DELAY = 300;

/**
 * Custom hook for debounced values
 * 
 * @param initialValue - Initial value
 * @param options - Debounce options
 * @returns Value, debounced value, and setter
 * 
 * @example
 * ```typescript
 * const { value, debouncedValue, setValue } = useDebouncedValue("", { delay: 500 });
 * 
 * <input value={value} onChange={(e) => setValue(e.target.value)} />
 * // debouncedValue updates 500ms after user stops typing
 * ```
 */
export function useDebouncedValue<T>(
  initialValue: T,
  options: UseDebouncedValueOptions = {}
): UseDebouncedValueReturn<T> {
  const { delay = DEFAULT_DELAY, immediate = false } = options;
  const [value, setValue] = useState<T>(initialValue);
  const [debouncedValue, setDebouncedValue] = useState<T>(initialValue);
  const [isDebouncing, setIsDebouncing] = useState(false);

  useEffect(() => {
    if (immediate && value === initialValue) {
      setDebouncedValue(value);
      return;
    }

    setIsDebouncing(true);
    const timer = setTimeout(() => {
      setDebouncedValue(value);
      setIsDebouncing(false);
    }, delay);

    return () => {
      clearTimeout(timer);
      setIsDebouncing(false);
    };
  }, [value, delay, immediate, initialValue]);

  return {
    value,
    debouncedValue,
    isDebouncing,
    setValue,
  };
}




