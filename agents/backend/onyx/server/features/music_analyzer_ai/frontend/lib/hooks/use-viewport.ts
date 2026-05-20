/**
 * Custom hook for viewport and window size detection.
 * Provides responsive breakpoint detection and window size tracking.
 */

import { useState, useEffect } from 'react';

/**
 * Breakpoint configuration.
 */
export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

/**
 * Viewport size interface.
 */
export interface ViewportSize {
  width: number;
  height: number;
}

/**
 * Breakpoint status interface.
 */
export interface BreakpointStatus {
  isSm: boolean;
  isMd: boolean;
  isLg: boolean;
  isXl: boolean;
  is2Xl: boolean;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
}

/**
 * Options for useViewport hook.
 */
export interface UseViewportOptions {
  debounceMs?: number;
}

/**
 * Custom hook for tracking viewport size and breakpoints.
 * @param options - Hook options
 * @returns Viewport size and breakpoint status
 */
export function useViewport(
  options: UseViewportOptions = {}
): ViewportSize & BreakpointStatus {
  const { debounceMs = 100 } = options;

  const [viewport, setViewport] = useState<ViewportSize>(() => {
    if (typeof window === 'undefined') {
      return { width: 0, height: 0 };
    }
    return {
      width: window.innerWidth,
      height: window.innerHeight,
    };
  });

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    let timeoutId: NodeJS.Timeout;

    const handleResize = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        setViewport({
          width: window.innerWidth,
          height: window.innerHeight,
        });
      }, debounceMs);
    };

    window.addEventListener('resize', handleResize, { passive: true });

    // Initial check
    handleResize();

    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener('resize', handleResize);
    };
  }, [debounceMs]);

  const breakpoints: BreakpointStatus = {
    isSm: viewport.width >= BREAKPOINTS.sm,
    isMd: viewport.width >= BREAKPOINTS.md,
    isLg: viewport.width >= BREAKPOINTS.lg,
    isXl: viewport.width >= BREAKPOINTS.xl,
    is2Xl: viewport.width >= BREAKPOINTS['2xl'],
    isMobile: viewport.width < BREAKPOINTS.md,
    isTablet:
      viewport.width >= BREAKPOINTS.md && viewport.width < BREAKPOINTS.lg,
    isDesktop: viewport.width >= BREAKPOINTS.lg,
  };

  return {
    ...viewport,
    ...breakpoints,
  };
}

