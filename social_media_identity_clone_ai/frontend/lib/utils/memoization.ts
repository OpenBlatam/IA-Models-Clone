import { useMemo, useCallback } from 'react';

export const createMemoizedSelector = <T, R>(
  selector: (state: T) => R,
  equalityFn?: (a: R, b: R) => boolean
) => {
  let lastResult: R | undefined;
  let lastState: T | undefined;

  return (state: T): R => {
    if (state === lastState) {
      return lastResult!;
    }

    const result = selector(state);
    
    if (equalityFn && lastResult !== undefined) {
      if (equalityFn(result, lastResult)) {
        return lastResult;
      }
    }

    lastState = state;
    lastResult = result;
    return result;
  };
};

export const useStableCallback = <T extends (...args: unknown[]) => unknown>(
  callback: T
): T => {
  return useCallback(callback, []) as T;
};

export const useStableMemo = <T>(factory: () => T, deps: React.DependencyList): T => {
  return useMemo(factory, deps);
};



