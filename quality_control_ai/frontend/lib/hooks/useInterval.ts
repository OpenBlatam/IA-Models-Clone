import { useEffect, useRef } from 'react';

interface UseIntervalOptions {
  enabled?: boolean;
  delay: number | null;
}

export const useInterval = (
  callback: () => void,
  { enabled = true, delay }: UseIntervalOptions
): void => {
  const savedCallback = useRef<() => void>();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!enabled || delay === null) {
      return;
    }

    const id = setInterval(() => {
      savedCallback.current?.();
    }, delay);

    return () => clearInterval(id);
  }, [enabled, delay]);
};

