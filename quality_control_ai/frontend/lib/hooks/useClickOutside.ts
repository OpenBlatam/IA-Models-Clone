import { useEffect, useRef, RefObject } from 'react';

export const useClickOutside = <T extends HTMLElement = HTMLDivElement>(
  handler: (event: MouseEvent | TouchEvent) => void
): RefObject<T> => {
  const ref = useRef<T>(null);

  useEffect(() => {
    const listener = (event: MouseEvent | TouchEvent): void => {
      const element = ref.current;
      if (!element || element.contains(event.target as Node)) {
        return;
      }
      handler(event);
    };

    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);

    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [handler]);

  return ref;
};
