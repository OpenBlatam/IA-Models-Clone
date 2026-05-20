/**
 * Custom hook for detecting clicks outside an element.
 * Useful for closing modals, dropdowns, and popovers.
 */

import { useEffect, useRef, type RefObject } from 'react';

/**
 * Options for click outside hook.
 */
export interface UseClickOutsideOptions {
  handler: (event: MouseEvent | TouchEvent) => void;
  enabled?: boolean;
}

/**
 * Custom hook for detecting clicks outside an element.
 * Triggers a handler when a click occurs outside the referenced element.
 *
 * @param options - Hook options
 * @returns Ref to attach to the element
 */
export function useClickOutside<T extends HTMLElement = HTMLElement>(
  options: UseClickOutsideOptions
): RefObject<T> {
  const { handler, enabled = true } = options;
  const ref = useRef<T>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const handleClickOutside = (event: MouseEvent | TouchEvent) => {
      const element = ref.current;
      if (!element) {
        return;
      }

      const target = event.target as Node;
      if (!element.contains(target)) {
        handler(event);
      }
    };

    // Use capture phase to catch events before they bubble
    document.addEventListener('mousedown', handleClickOutside, true);
    document.addEventListener('touchstart', handleClickOutside, true);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside, true);
      document.removeEventListener('touchstart', handleClickOutside, true);
    };
  }, [handler, enabled]);

  return ref;
}

