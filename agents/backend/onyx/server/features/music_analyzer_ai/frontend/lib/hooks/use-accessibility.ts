/**
 * Accessibility hooks.
 * Provides React hooks for accessibility features.
 */

import { useEffect, useRef, useCallback } from 'react';
import {
  announceToScreenReader,
  AriaLivePriority,
  trapFocus,
  restoreFocus,
  saveFocus,
  prefersReducedMotion,
  prefersHighContrast,
} from '@/lib/utils/accessibility';

/**
 * Options for useAnnounce hook.
 */
export interface UseAnnounceOptions {
  /**
   * Priority of the announcement.
   */
  priority?: AriaLivePriority;
  /**
   * Whether to announce immediately on mount.
   */
  announceOnMount?: boolean;
}

/**
 * Hook for announcing messages to screen readers.
 *
 * @param message - Message to announce
 * @param options - Announce options
 * @returns Function to trigger announcement
 *
 * @example
 * ```tsx
 * const announce = useAnnounce();
 * announce('Form submitted successfully');
 * ```
 */
export function useAnnounce(
  initialMessage?: string,
  options?: UseAnnounceOptions
): (message: string, priority?: AriaLivePriority) => void {
  const { priority = AriaLivePriority.POLITE, announceOnMount = false } =
    options || {};

  const announce = useCallback(
    (message: string, msgPriority?: AriaLivePriority) => {
      announceToScreenReader(message, msgPriority || priority);
    },
    [priority]
  );

  useEffect(() => {
    if (announceOnMount && initialMessage) {
      announce(initialMessage);
    }
  }, [announceOnMount, initialMessage, announce]);

  return announce;
}

/**
 * Options for useFocusTrap hook.
 */
export interface UseFocusTrapOptions {
  /**
   * Whether the trap is active.
   */
  active?: boolean;
  /**
   * Element to trap focus within.
   */
  containerRef?: React.RefObject<HTMLElement>;
  /**
   * Whether to restore focus on unmount.
   */
  restoreFocusOnUnmount?: boolean;
}

/**
 * Hook for trapping focus within a container.
 *
 * @param options - Focus trap options
 * @returns Ref to attach to container
 *
 * @example
 * ```tsx
 * const containerRef = useFocusTrap({ active: isOpen });
 * return <div ref={containerRef}>...</div>;
 * ```
 */
export function useFocusTrap(
  options: UseFocusTrapOptions = {}
): React.RefObject<HTMLElement> {
  const {
    active = true,
    containerRef: externalRef,
    restoreFocusOnUnmount = true,
  } = options;

  const internalRef = useRef<HTMLElement>(null);
  const containerRef = externalRef || internalRef;
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!active || !containerRef.current) {
      return;
    }

    // Save current focus
    previousFocusRef.current = saveFocus();

    // Trap focus
    cleanupRef.current = trapFocus(containerRef.current);

    return () => {
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }

      // Restore focus
      if (restoreFocusOnUnmount && previousFocusRef.current) {
        restoreFocus(previousFocusRef.current);
      }
    };
  }, [active, containerRef, restoreFocusOnUnmount]);

  return containerRef;
}

/**
 * Hook for detecting reduced motion preference.
 *
 * @returns Whether reduced motion is preferred
 *
 * @example
 * ```tsx
 * const prefersReduced = usePrefersReducedMotion();
 * const animationClass = prefersReduced ? 'no-animation' : 'animate';
 * ```
 */
export function usePrefersReducedMotion(): boolean {
  const [prefersReduced, setPrefersReduced] = React.useState(() =>
    prefersReducedMotion()
  );

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handleChange = () => setPrefersReduced(mediaQuery.matches);

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  return prefersReduced;
}

/**
 * Hook for detecting high contrast preference.
 *
 * @returns Whether high contrast is preferred
 *
 * @example
 * ```tsx
 * const prefersHighContrast = usePrefersHighContrast();
 * const theme = prefersHighContrast ? 'high-contrast' : 'default';
 * ```
 */
export function usePrefersHighContrast(): boolean {
  const [prefersHighContrast, setPrefersHighContrast] = React.useState(() =>
    prefersHighContrast()
  );

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const mediaQueries = [
      window.matchMedia('(prefers-contrast: high)'),
      window.matchMedia('(prefers-contrast: more)'),
    ];

    const handleChange = () => {
      setPrefersHighContrast(
        mediaQueries.some((mq) => mq.matches) || prefersHighContrast()
      );
    };

    mediaQueries.forEach((mq) => {
      mq.addEventListener('change', handleChange);
    });

    return () => {
      mediaQueries.forEach((mq) => {
        mq.removeEventListener('change', handleChange);
      });
    };
  }, []);

  return prefersHighContrast;
}

/**
 * Hook for keyboard navigation.
 *
 * @param handlers - Keyboard event handlers
 * @returns Props to spread on element
 *
 * @example
 * ```tsx
 * const keyboardProps = useKeyboardNavigation({
 *   onEnter: () => handleSubmit(),
 *   onEscape: () => handleClose(),
 * });
 * return <div {...keyboardProps}>...</div>;
 * ```
 */
export function useKeyboardNavigation(handlers: {
  onEnter?: () => void;
  onEscape?: () => void;
  onArrowUp?: () => void;
  onArrowDown?: () => void;
  onArrowLeft?: () => void;
  onArrowRight?: () => void;
  onTab?: () => void;
  onSpace?: () => void;
}): {
  onKeyDown: (e: React.KeyboardEvent) => void;
  tabIndex: number;
} {
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'Enter':
          handlers.onEnter?.();
          break;
        case 'Escape':
          handlers.onEscape?.();
          break;
        case 'ArrowUp':
          handlers.onArrowUp?.();
          e.preventDefault();
          break;
        case 'ArrowDown':
          handlers.onArrowDown?.();
          e.preventDefault();
          break;
        case 'ArrowLeft':
          handlers.onArrowLeft?.();
          e.preventDefault();
          break;
        case 'ArrowRight':
          handlers.onArrowRight?.();
          e.preventDefault();
          break;
        case 'Tab':
          handlers.onTab?.();
          break;
        case ' ':
          handlers.onSpace?.();
          e.preventDefault();
          break;
      }
    },
    [handlers]
  );

  return {
    onKeyDown: handleKeyDown,
    tabIndex: 0,
  };
}

// Import React
import React from 'react';




