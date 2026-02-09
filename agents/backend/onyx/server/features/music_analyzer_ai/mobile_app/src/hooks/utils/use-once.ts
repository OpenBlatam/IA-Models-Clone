import { useRef, useCallback } from 'react';

export function useOnce(callback: () => void) {
  const hasRun = useRef(false);

  const run = useCallback(() => {
    if (!hasRun.current) {
      hasRun.current = true;
      callback();
    }
  }, [callback]);

  return run;
}


