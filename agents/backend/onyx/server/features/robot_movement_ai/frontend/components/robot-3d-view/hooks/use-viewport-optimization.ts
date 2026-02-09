/**
 * Hook for viewport optimization
 * @module robot-3d-view/hooks/use-viewport-optimization
 */

import { useEffect, useState, useCallback } from 'react';

/**
 * Viewport size
 */
interface ViewportSize {
  width: number;
  height: number;
}

/**
 * Options for viewport optimization
 */
interface ViewportOptions {
  /** Debounce delay for resize events */
  debounceDelay?: number;
  /** Minimum viewport width */
  minWidth?: number;
  /** Minimum viewport height */
  minHeight?: number;
}

/**
 * Hook for optimizing 3D view based on viewport size
 * 
 * Automatically adjusts quality and settings based on viewport dimensions.
 * 
 * @param options - Viewport optimization options
 * @returns Viewport size and optimization settings
 * 
 * @example
 * ```tsx
 * const { viewport, isMobile, quality } = useViewportOptimization();
 * ```
 */
export function useViewportOptimization(options: ViewportOptions = {}) {
  const { debounceDelay = 150, minWidth = 320, minHeight = 240 } = options;
  const [viewport, setViewport] = useState<ViewportSize>(() => {
    if (typeof window === 'undefined') {
      return { width: 1920, height: 1080 };
    }
    return {
      width: Math.max(window.innerWidth, minWidth),
      height: Math.max(window.innerHeight, minHeight),
    };
  });

  const handleResize = useCallback(() => {
    if (typeof window === 'undefined') return;

    setViewport({
      width: Math.max(window.innerWidth, minWidth),
      height: Math.max(window.innerHeight, minHeight),
    });
  }, [minWidth, minHeight]);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    let timeoutId: NodeJS.Timeout;
    const debouncedResize = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(handleResize, debounceDelay);
    };

    window.addEventListener('resize', debouncedResize);
    handleResize(); // Initial call

    return () => {
      window.removeEventListener('resize', debouncedResize);
      clearTimeout(timeoutId);
    };
  }, [handleResize, debounceDelay]);

  const isMobile = viewport.width < 768;
  const isTablet = viewport.width >= 768 && viewport.width < 1024;
  const isDesktop = viewport.width >= 1024;

  // Calculate quality based on viewport
  const quality = {
    dpr: isMobile ? [1, 1.5] as [number, number] : [1, 2] as [number, number],
    shadows: !isMobile,
    antialias: !isMobile,
    particleCount: isMobile ? 10 : isTablet ? 15 : 20,
    gridSize: isMobile ? 8 : 10,
    enablePostProcessing: isDesktop,
  };

  return {
    viewport,
    isMobile,
    isTablet,
    isDesktop,
    quality,
  };
}



