/**
 * Performance utilities for Continuous Agent feature
 */
import { throttle as lodashThrottle } from "lodash-es";

export const throttle = <T extends (...args: unknown[]) => unknown>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  return lodashThrottle(func, limit, { leading: true, trailing: false });
};

export const requestIdleCallback = (
  callback: () => void,
  options?: { timeout?: number }
): number => {
  if (typeof window !== "undefined" && "requestIdleCallback" in window) {
    return window.requestIdleCallback(callback, options);
  }
  return setTimeout(callback, options?.timeout || 1) as unknown as number;
};

export const cancelIdleCallback = (id: number): void => {
  if (typeof window !== "undefined" && "cancelIdleCallback" in window) {
    window.cancelIdleCallback(id);
  } else {
    clearTimeout(id);
  }
};

export const measurePerformance = (label: string): (() => void) => {
  if (typeof window !== "undefined" && "performance" in window) {
    const start = performance.now();
    return () => {
      const end = performance.now();
      console.log(`${label}: ${end - start}ms`);
    };
  }
  return () => {};
};



