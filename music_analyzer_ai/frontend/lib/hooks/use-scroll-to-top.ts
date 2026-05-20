/**
 * Custom hook for scroll to top functionality.
 * Provides smooth scroll to top with visibility control.
 */

import { useState, useEffect, useCallback } from 'react';
import { useScroll } from './use-scroll';

/**
 * Options for useScrollToTop hook.
 */
export interface UseScrollToTopOptions {
  threshold?: number;
  smooth?: boolean;
}

/**
 * Return type for useScrollToTop hook.
 */
export interface UseScrollToTopReturn {
  isVisible: boolean;
  scrollToTop: () => void;
}

/**
 * Custom hook for scroll to top button.
 * Shows button when scrolled down beyond threshold.
 *
 * @param options - Hook options
 * @returns Scroll to top state and handler
 */
export function useScrollToTop(
  options: UseScrollToTopOptions = {}
): UseScrollToTopReturn {
  const { threshold = 400, smooth = true } = options;
  const { position } = useScroll();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(position.y > threshold);
  }, [position.y, threshold]);

  const scrollToTop = useCallback(() => {
    if (smooth) {
      window.scrollTo({
        top: 0,
        behavior: 'smooth',
      });
    } else {
      window.scrollTo(0, 0);
    }
  }, [smooth]);

  return {
    isVisible,
    scrollToTop,
  };
}

