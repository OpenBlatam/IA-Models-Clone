/**
 * Custom hook for tooltip state management.
 * Provides convenient methods for showing/hiding tooltips programmatically.
 */

import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * Options for useTooltip hook.
 */
export interface UseTooltipOptions {
  delay?: number;
  duration?: number;
}

/**
 * Return type for useTooltip hook.
 */
export interface UseTooltipReturn {
  isVisible: boolean;
  show: () => void;
  hide: () => void;
  toggle: () => void;
}

/**
 * Custom hook for tooltip state management.
 * Provides programmatic control over tooltip visibility.
 *
 * @param options - Hook options
 * @returns Tooltip state and handlers
 */
export function useTooltip(
  options: UseTooltipOptions = {}
): UseTooltipReturn {
  const { delay = 200, duration } = options;
  const [isVisible, setIsVisible] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const show = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);

      if (duration) {
        hideTimeoutRef.current = setTimeout(() => {
          setIsVisible(false);
        }, duration);
      }
    }, delay);
  }, [delay, duration]);

  const hide = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
    }
    setIsVisible(false);
  }, []);

  const toggle = useCallback(() => {
    if (isVisible) {
      hide();
    } else {
      show();
    }
  }, [isVisible, show, hide]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
      }
    };
  }, []);

  return {
    isVisible,
    show,
    hide,
    toggle,
  };
}

