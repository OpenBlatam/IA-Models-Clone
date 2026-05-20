import { useEffect, useState } from 'react';

interface UseIdleOptions {
  timeout?: number;
  events?: string[];
  initialState?: boolean;
}

const DEFAULT_TIMEOUT = 3000;
const DEFAULT_EVENTS = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];

export const useIdle = (options: UseIdleOptions = {}): boolean => {
  const { timeout = DEFAULT_TIMEOUT, events = DEFAULT_EVENTS, initialState = false } = options;
  const [isIdle, setIsIdle] = useState(initialState);

  useEffect(() => {
    let idleTimer: NodeJS.Timeout;

    const handleEvent = (): void => {
      setIsIdle(false);
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => setIsIdle(true), timeout);
    };

    events.forEach((event) => {
      document.addEventListener(event, handleEvent, { passive: true });
    });

    idleTimer = setTimeout(() => setIsIdle(true), timeout);

    return () => {
      clearTimeout(idleTimer);
      events.forEach((event) => {
        document.removeEventListener(event, handleEvent);
      });
    };
  }, [timeout, events]);

  return isIdle;
};



