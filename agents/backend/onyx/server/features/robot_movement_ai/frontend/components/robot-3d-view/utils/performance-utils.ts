/**
 * Performance utilities for 3D rendering
 * @module robot-3d-view/utils/performance-utils
 */

/**
 * Throttles function execution for performance
 * 
 * @param func - Function to throttle
 * @param delay - Delay in milliseconds
 * @returns Throttled function
 */
export function throttle<T extends (...args: any[]) => any>(func: T, delay: number): T {
  let lastCall = 0;
  return ((...args: Parameters<T>) => {
    const now = Date.now();
    if (now - lastCall >= delay) {
      lastCall = now;
      return func(...args);
    }
  }) as T;
}

/**
 * Debounces function execution
 * 
 * @param func - Function to debounce
 * @param delay - Delay in milliseconds
 * @returns Debounced function
 */
export function debounce<T extends (...args: any[]) => any>(func: T, delay: number): T {
  let timeoutId: NodeJS.Timeout;
  return ((...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  }) as T;
}

/**
 * Checks if device has low performance
 * 
 * @returns true if device is considered low performance
 */
export function isLowPerformanceDevice(): boolean {
  if (typeof window === 'undefined') return false;

  // Check for mobile devices
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );

  // Check for low-end GPU
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
  
  if (!gl) return true;

  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  if (debugInfo) {
    const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    // Check for integrated graphics
    const isIntegrated = /Intel|Integrated/i.test(renderer);
    return isMobile || isIntegrated;
  }

  return isMobile;
}

/**
 * Gets optimal quality settings based on device
 * 
 * @returns Quality configuration
 */
export function getOptimalQuality() {
  const isLowPerf = isLowPerformanceDevice();

  return {
    dpr: isLowPerf ? [1, 1.5] as [number, number] : [1, 2] as [number, number],
    shadows: !isLowPerf,
    antialias: !isLowPerf,
    particleCount: isLowPerf ? 10 : 20,
    gridSize: isLowPerf ? 8 : 10,
  };
}



