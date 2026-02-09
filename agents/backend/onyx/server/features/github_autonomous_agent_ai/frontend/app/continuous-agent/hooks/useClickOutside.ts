/**
 * Custom hook to detect clicks outside an element
 * 
 * Useful for closing modals, dropdowns, etc.
 */

import { useEffect, useRef, type RefObject } from "react";

/**
 * Options for useClickOutside hook
 */
export interface UseClickOutsideOptions {
  /** Whether the hook is enabled */
  readonly enabled?: boolean;
  /** Event type to listen to (default: 'mousedown') */
  readonly event?: "mousedown" | "mouseup" | "click";
}

/**
 * Custom hook to detect clicks outside an element
 * 
 * @param handler - Callback function when click outside is detected
 * @param options - Hook options
 * @returns Ref to attach to the element
 * 
 * @example
 * ```typescript
 * const ref = useClickOutside(() => {
 *   setIsOpen(false);
 * });
 * 
 * return <div ref={ref}>Content</div>;
 * ```
 */
export function useClickOutside<T extends HTMLElement = HTMLElement>(
  handler: () => void,
  options: UseClickOutsideOptions = {}
): RefObject<T> {
  const { enabled = true, event = "mousedown" } = options;
  const ref = useRef<T>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const handleClickOutside = (event: MouseEvent): void => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        handler();
      }
    };

    document.addEventListener(event, handleClickOutside);
    return () => {
      document.removeEventListener(event, handleClickOutside);
    };
  }, [handler, enabled, event]);

  return ref;
}




