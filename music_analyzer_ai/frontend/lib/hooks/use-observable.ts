/**
 * Custom hook for observable.
 * Provides reactive observable functionality.
 */

import { useState, useEffect, useRef } from 'react';
import { Observable, Observer } from '../utils/observable';

/**
 * Custom hook for observable.
 * Provides reactive observable functionality.
 *
 * @param observable - Observable instance
 * @returns Current observable value
 */
export function useObservable<T>(observable: Observable<T>): T {
  const [value, setValue] = useState<T>(observable.value);

  useEffect(() => {
    const unsubscribe = observable.subscribe((newValue) => {
      setValue(newValue);
    });

    return unsubscribe;
  }, [observable]);

  return value;
}

/**
 * Custom hook for creating observable.
 * Creates and manages observable instance.
 *
 * @param initialValue - Initial observable value
 * @returns Observable instance and current value
 */
export function useObservableState<T>(initialValue: T) {
  const observableRef = useRef<Observable<T>>(new Observable(initialValue));
  const value = useObservable(observableRef.current);

  const setValue = (newValue: T | ((prev: T) => T)) => {
    if (typeof newValue === 'function') {
      observableRef.current.value = (newValue as (prev: T) => T)(
        observableRef.current.value
      );
    } else {
      observableRef.current.value = newValue;
    }
  };

  return [value, setValue, observableRef.current] as const;
}

