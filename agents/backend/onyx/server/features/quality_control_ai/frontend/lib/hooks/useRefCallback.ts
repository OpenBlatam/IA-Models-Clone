import { useRef, useCallback } from 'react';

export const useRefCallback = <T extends (...args: unknown[]) => unknown>(
  callback: T
): [React.MutableRefObject<T>, T] => {
  const callbackRef = useRef(callback);

  const stableCallback = useCallback(
    ((...args: Parameters<T>) => {
      return callbackRef.current(...args);
    }) as T,
    []
  );

  callbackRef.current = callback;

  return [callbackRef, stableCallback];
};

