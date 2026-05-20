/**
 * Custom hook for state with history.
 * Provides undo/redo functionality for state.
 */

import { useState, useCallback, useRef } from 'react';

/**
 * Options for useStateWithHistory hook.
 */
export interface UseStateWithHistoryOptions {
  maxHistory?: number;
}

/**
 * Custom hook for state with history.
 * Provides undo/redo functionality for state.
 *
 * @param initialValue - Initial state value
 * @param options - History options
 * @returns State with history operations
 */
export function useStateWithHistory<T>(
  initialValue: T,
  options: UseStateWithHistoryOptions = {}
) {
  const { maxHistory = 50 } = options;
  const [state, setState] = useState<T>(initialValue);
  const historyRef = useRef<T[]>([initialValue]);
  const currentIndexRef = useRef(0);

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      const newValue = typeof value === 'function' ? (value as (prev: T) => T)(state) : value;

      // Remove any future history if we're not at the end
      historyRef.current = historyRef.current.slice(0, currentIndexRef.current + 1);
      historyRef.current.push(newValue);

      // Limit history size
      if (historyRef.current.length > maxHistory) {
        historyRef.current.shift();
      } else {
        currentIndexRef.current++;
      }

      setState(newValue);
    },
    [state, maxHistory]
  );

  const undo = useCallback(() => {
    if (currentIndexRef.current > 0) {
      currentIndexRef.current--;
      setState(historyRef.current[currentIndexRef.current]);
    }
  }, []);

  const redo = useCallback(() => {
    if (currentIndexRef.current < historyRef.current.length - 1) {
      currentIndexRef.current++;
      setState(historyRef.current[currentIndexRef.current]);
    }
  }, []);

  const canUndo = currentIndexRef.current > 0;
  const canRedo = currentIndexRef.current < historyRef.current.length - 1;

  return {
    state,
    setState: setValue,
    undo,
    redo,
    canUndo,
    canRedo,
    history: historyRef.current,
    currentIndex: currentIndexRef.current,
  };
}

