/**
 * Custom hook for scroll position detection.
 * Provides reactive scroll position and scroll direction.
 */

import { useState, useEffect } from 'react';
import { throttle } from '../utils/performance';

/**
 * Scroll position interface.
 */
export interface ScrollPosition {
  x: number;
  y: number;
}

/**
 * Options for useScroll hook.
 */
export interface UseScrollOptions {
  throttleMs?: number;
  element?: HTMLElement | null;
}

/**
 * Custom hook for tracking scroll position.
 * Throttled for performance optimization.
 *
 * @param options - Hook options
 * @returns Scroll position and direction
 */
export function useScroll(options: UseScrollOptions = {}): {
  position: ScrollPosition;
  direction: 'up' | 'down' | null;
  isAtTop: boolean;
  isAtBottom: boolean;
} {
  const { throttleMs = 100, element } = options;

  const [position, setPosition] = useState<ScrollPosition>({ x: 0, y: 0 });
  const [direction, setDirection] = useState<'up' | 'down' | null>(null);
  const [isAtTop, setIsAtTop] = useState(true);
  const [isAtBottom, setIsAtBottom] = useState(false);

  useEffect(() => {
    const targetElement = element || (typeof window !== 'undefined' ? window : null);
    
    if (!targetElement) {
      return;
    }

    let lastY = 0;

    const handleScroll = throttle(() => {
      const scrollY = 'scrollY' in targetElement 
        ? (targetElement as Window).scrollY 
        : (targetElement as HTMLElement).scrollTop;
      
      const scrollX = 'scrollX' in targetElement
        ? (targetElement as Window).scrollX
        : (targetElement as HTMLElement).scrollLeft;

      const maxScroll = 'innerHeight' in targetElement
        ? document.documentElement.scrollHeight - window.innerHeight
        : (targetElement as HTMLElement).scrollHeight - (targetElement as HTMLElement).clientHeight;

      setPosition({ x: scrollX, y: scrollY });
      setDirection(scrollY > lastY ? 'down' : scrollY < lastY ? 'up' : null);
      setIsAtTop(scrollY <= 0);
      setIsAtBottom(scrollY >= maxScroll - 1);

      lastY = scrollY;
    }, throttleMs);

    const scrollTarget = 'addEventListener' in targetElement 
      ? targetElement 
      : window;

    scrollTarget.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll(); // Initial call

    return () => {
      scrollTarget.removeEventListener('scroll', handleScroll);
    };
  }, [throttleMs, element]);

  return {
    position,
    direction,
    isAtTop,
    isAtBottom,
  };
}

