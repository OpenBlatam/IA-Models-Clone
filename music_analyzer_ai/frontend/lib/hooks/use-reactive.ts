/**
 * Custom hook for reactive values.
 * Provides reactive value functionality.
 */

import { useState, useEffect, useRef } from 'react';
import { Reactive, reactive, computed } from '../utils/reactive';

/**
 * Custom hook for reactive values.
 * Provides reactive value functionality.
 *
 * @param initialValue - Initial reactive value
 * @returns Reactive value and setter
 */
export function useReactive<T>(initialValue: T) {
  const reactiveRef = useRef<Reactive<T>>(reactive(initialValue));
  const [value, setValue] = useState<T>(initialValue);

  useEffect(() => {
    const unsubscribe = reactiveRef.current.subscribe((newValue) => {
      setValue(newValue);
    });

    return unsubscribe;
  }, []);

  const setReactiveValue = (newValue: T | ((prev: T) => T)) => {
    if (typeof newValue === 'function') {
      reactiveRef.current.update(newValue as (prev: T) => T);
    } else {
      reactiveRef.current.value = newValue;
    }
  };

  return [value, setReactiveValue, reactiveRef.current] as const;
}

/**
 * Custom hook for computed reactive values.
 * Provides computed reactive value functionality.
 *
 * @param source - Source reactive value
 * @param fn - Computation function
 * @returns Computed reactive value
 */
export function useComputed<T, R>(
  source: Reactive<T>,
  fn: (value: T) => R
): R {
  const computedRef = useRef<Reactive<R>>(computed(source, fn));
  const [value, setValue] = useState<R>(computedRef.current.value);

  useEffect(() => {
    const unsubscribe = computedRef.current.subscribe((newValue) => {
      setValue(newValue);
    });

    return unsubscribe;
  }, [source, fn]);

  return value;
}

