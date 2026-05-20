import { useEffect, useRef, useCallback, useState } from 'react';

export const useTimeout = (callback: () => void, delay: number | null): void => {
  const savedCallback = useRef(callback);
  const timeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay === null) return;

    timeoutRef.current = setTimeout(() => {
      savedCallback.current();
    }, delay);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [delay]);
};

export const useTimeoutFn = (
  callback: () => void,
  delay: number | null
): [() => void, () => void, boolean] => {
  const timeoutRef = useRef<NodeJS.Timeout>();
  const callbackRef = useRef(callback);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  const set = useCallback(() => {
    if (delay === null) return;
    setIsReady(false);
    timeoutRef.current = setTimeout(() => {
      callbackRef.current();
      setIsReady(true);
    }, delay);
  }, [delay]);

  const clear = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      setIsReady(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      clear();
    };
  }, [clear]);

  return [set, clear, isReady];
};

