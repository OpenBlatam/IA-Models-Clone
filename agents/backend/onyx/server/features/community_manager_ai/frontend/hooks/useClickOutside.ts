/**
 * useClickOutside Hook
 * Hook to detect clicks outside of a referenced element
 */

import { useEffect, RefObject } from 'react';

/**
 * Hook to detect clicks outside of a referenced element
 * @param ref - Ref object for the element
 * @param handler - Callback function when click outside is detected
 * @param enabled - Whether the hook is enabled (default: true)
 */
export const useClickOutside = <T extends HTMLElement = HTMLElement>(
  ref: RefObject<T>,
  handler: (event: MouseEvent | TouchEvent) => void,
  enabled: boolean = true
) => {
  useEffect(() => {
    if (!enabled) return;

    const listener = (event: MouseEvent | TouchEvent) => {
      const el = ref?.current;
      if (!el || el.contains(event.target as Node)) {
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
  }, [ref, handler, enabled]);
};
