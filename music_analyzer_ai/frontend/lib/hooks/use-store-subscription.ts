/**
 * Custom hook for subscribing to store changes.
 * Useful for side effects when store state changes.
 */

import { useEffect, useRef } from 'react';
import type { StoreApi, UseBoundStore } from 'zustand';

/**
 * Subscribes to store changes and runs effect.
 * @param store - Zustand store
 * @param selector - Selector function
 * @param effect - Effect function to run on changes
 * @param deps - Dependency array (optional)
 */
export function useStoreSubscription<T, U>(
  store: UseBoundStore<StoreApi<T>>,
  selector: (state: T) => U,
  effect: (selected: U, previous: U | undefined) => void | (() => void),
  deps?: unknown[]
): void {
  const previousRef = useRef<U>();

  useEffect(() => {
    const unsubscribe = store.subscribe((state) => {
      const selected = selector(state);
      const previous = previousRef.current;

      if (selected !== previous) {
        const cleanup = effect(selected, previous);
        previousRef.current = selected;
        return cleanup;
      }
    });

    // Initial call
    const initial = selector(store.getState());
    const cleanup = effect(initial, undefined);
    previousRef.current = initial;

    return () => {
      unsubscribe();
      if (typeof cleanup === 'function') {
        cleanup();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

