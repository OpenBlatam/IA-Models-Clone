/**
 * Math utilities
 */

/**
 * Clamp number between min and max
 */
export const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max);
};

/**
 * Linear interpolation
 */
export const lerp = (start: number, end: number, t: number): number => {
  return start + (end - start) * t;
};

/**
 * Map value from one range to another
 */
export const mapRange = (
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number => {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
};

/**
 * Normalize value to 0-1 range
 */
export const normalize = (value: number, min: number, max: number): number => {
  return (value - min) / (max - min);
};

/**
 * Easing functions
 */
export const easing = {
  linear: (t: number) => t,
  easeIn: (t: number) => t * t,
  easeOut: (t: number) => t * (2 - t),
  easeInOut: (t: number) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
  easeInQuad: (t: number) => t * t,
  easeOutQuad: (t: number) => t * (2 - t),
  easeInOutQuad: (t: number) =>
    t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeInCubic: (t: number) => t * t * t,
  easeOutCubic: (t: number) => --t * t * t + 1,
  easeInOutCubic: (t: number) =>
    t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
};

/**
 * Calculate distance between two points
 */
export const distance = (
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number => {
  return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
};

/**
 * Calculate angle between two points
 */
export const angle = (
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number => {
  return Math.atan2(y2 - y1, x2 - x1);
};

/**
 * Convert degrees to radians
 */
export const degToRad = (degrees: number): number => {
  return (degrees * Math.PI) / 180;
};

/**
 * Convert radians to degrees
 */
export const radToDeg = (radians: number): number => {
  return (radians * 180) / Math.PI;
};

/**
 * Round to nearest multiple
 */
export const roundToNearest = (value: number, multiple: number): number => {
  return Math.round(value / multiple) * multiple;
};

/**
 * Check if number is even
 */
export const isEven = (n: number): boolean => {
  return n % 2 === 0;
};

/**
 * Check if number is odd
 */
export const isOdd = (n: number): boolean => {
  return n % 2 !== 0;
};

