'use client';

import { useState, useEffect, useRef } from 'react';

interface UseIdleOptions {
  timeout?: number;
  events?: string[];
  initialState?: boolean;
}

const DEFAULT_TIMEOUT = 3000; // 3 seconds
const DEFAULT_EVENTS = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];

export function useIdle(options: UseIdleOptions = {}) {
  const {
    timeout = DEFAULT_TIMEOUT,
    events = DEFAULT_EVENTS,
    initialState = false,
  } = options;

  const [isIdle, setIsIdle] = useState(initialState);
  const timeoutIdRef = useRef<NodeJS.Timeout | undefined>();

  useEffect(() => {
    const handleEvent = () => {
      setIsIdle(false);

      if (timeoutIdRef.current) {
        clearTimeout(timeoutIdRef.current);
      }

      timeoutIdRef.current = setTimeout(() => {
        setIsIdle(true);
      }, timeout);
    };

    events.forEach((event) => {
      document.addEventListener(event, handleEvent, { passive: true });
    });

    // Initial timeout
    timeoutIdRef.current = setTimeout(() => {
      setIsIdle(true);
    }, timeout);

    return () => {
      events.forEach((event) => {
        document.removeEventListener(event, handleEvent);
      });
      if (timeoutIdRef.current) {
        clearTimeout(timeoutIdRef.current);
      }
    };
  }, [timeout, events]);

  return isIdle;
}
