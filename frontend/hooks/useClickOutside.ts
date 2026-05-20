'use client';

import { useEffect, RefObject, useRef, useCallback } from 'react';

export function useClickOutside<T extends HTMLElement>(
  handler: (event: MouseEvent | TouchEvent) => void
): RefObject<T> {
  const ref = useRef<T>(null);

  const listener = useCallback(
    (event: MouseEvent | TouchEvent) => {
      if (!ref.current || ref.current.contains(event.target as Node)) {
        return;
      }
      handler(event);
    },
    [handler]
  );

  useEffect(() => {
    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);

    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [listener]);

  return ref;
}

