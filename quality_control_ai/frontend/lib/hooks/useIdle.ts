import { useState, useEffect, useRef } from 'react';

interface UseIdleOptions {
  timeout?: number;
  events?: string[];
}

export const useIdle = ({ timeout = 30000, events = ['mousedown', 'keydown'] }: UseIdleOptions = {}): boolean => {
  const [isIdle, setIsIdle] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    const resetTimer = (): void => {
      setIsIdle(false);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        setIsIdle(true);
      }, timeout);
    };

    resetTimer();

    events.forEach((event) => {
      window.addEventListener(event, resetTimer);
    });

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      events.forEach((event) => {
        window.removeEventListener(event, resetTimer);
      });
    };
  }, [timeout, events]);

  return isIdle;
};

