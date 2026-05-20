/**
 * Gesture utilities
 */

/**
 * Calculate swipe direction
 */
export const getSwipeDirection = (
  dx: number,
  dy: number,
  threshold: number = 50
): 'left' | 'right' | 'up' | 'down' | null => {
  if (Math.abs(dx) > Math.abs(dy)) {
    if (Math.abs(dx) > threshold) {
      return dx > 0 ? 'right' : 'left';
    }
  } else {
    if (Math.abs(dy) > threshold) {
      return dy > 0 ? 'down' : 'up';
    }
  }
  return null;
};

/**
 * Calculate swipe velocity
 */
export const getSwipeVelocity = (
  dx: number,
  dy: number,
  time: number
): number => {
  const distance = Math.sqrt(dx * dx + dy * dy);
  return distance / time;
};

/**
 * Check if gesture is a tap
 */
export const isTap = (
  dx: number,
  dy: number,
  threshold: number = 10
): boolean => {
  return Math.abs(dx) < threshold && Math.abs(dy) < threshold;
};

/**
 * Check if gesture is a long press
 */
export const isLongPress = (duration: number, threshold: number = 500): boolean => {
  return duration > threshold;
};

/**
 * Calculate pinch scale
 */
export const calculatePinchScale = (
  distance1: number,
  distance2: number
): number => {
  if (distance1 === 0) return 1;
  return distance2 / distance1;
};

/**
 * Calculate rotation angle
 */
export const calculateRotation = (
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  centerX: number,
  centerY: number
): number => {
  const angle1 = Math.atan2(y1 - centerY, x1 - centerX);
  const angle2 = Math.atan2(y2 - centerY, x2 - centerX);
  return angle2 - angle1;
};

