/**
 * Custom hook for hover state detection.
 * Provides reactive hover state for elements.
 */

import { useState, useRef, useEffect, type RefObject } from 'react';

/**
 * Return type for useHover hook.
 */
export interface UseHoverReturn {
  isHovered: boolean;
  ref: RefObject<HTMLElement>;
}

/**
 * Custom hook for detecting hover state.
 * Tracks when an element is being hovered.
 *
 * @returns Hover state and ref
 */
export function useHover(): UseHoverReturn {
  const [isHovered, setIsHovered] = useState(false);
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    const handleMouseEnter = () => setIsHovered(true);
    const handleMouseLeave = () => setIsHovered(false);

    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      element.removeEventListener('mouseenter', handleMouseEnter);
      element.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  return { isHovered, ref };
}

