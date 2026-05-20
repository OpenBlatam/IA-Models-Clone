/**
 * Custom hook for focus trap functionality.
 * Traps focus within a specific element (useful for modals, dialogs).
 */

import { useEffect, useRef } from 'react';

/**
 * Options for useFocusTrap hook.
 */
export interface UseFocusTrapOptions {
  enabled?: boolean;
  initialFocus?: HTMLElement | null;
}

/**
 * Custom hook for focus trap.
 * Traps keyboard focus within a container element.
 *
 * @param options - Hook options
 * @returns Ref to attach to the container element
 */
export function useFocusTrap(
  options: UseFocusTrapOptions = {}
): React.RefObject<HTMLElement> {
  const { enabled = true, initialFocus } = options;
  const containerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!enabled || !containerRef.current) {
      return;
    }

    const container = containerRef.current;

    // Get all focusable elements
    const getFocusableElements = (): HTMLElement[] => {
      const selector = [
        'a[href]',
        'button:not([disabled])',
        'textarea:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        '[tabindex]:not([tabindex="-1"])',
      ].join(', ');

      return Array.from(container.querySelectorAll<HTMLElement>(selector)).filter(
        (el) => {
          const style = window.getComputedStyle(el);
          return style.display !== 'none' && style.visibility !== 'hidden';
        }
      );
    };

    const focusableElements = getFocusableElements();

    if (focusableElements.length === 0) {
      return;
    }

    // Focus initial element or first focusable element
    if (initialFocus && focusableElements.includes(initialFocus)) {
      initialFocus.focus();
    } else {
      focusableElements[0]?.focus();
    }

    const handleTabKey = (e: KeyboardEvent): void => {
      if (e.key !== 'Tab') {
        return;
      }

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey) {
        // Shift + Tab
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        // Tab
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    container.addEventListener('keydown', handleTabKey);

    return () => {
      container.removeEventListener('keydown', handleTabKey);
    };
  }, [enabled, initialFocus]);

  return containerRef;
}

