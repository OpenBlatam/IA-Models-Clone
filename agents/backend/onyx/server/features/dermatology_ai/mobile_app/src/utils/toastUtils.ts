/**
 * Toast utilities
 */

export type ToastType = 'success' | 'error' | 'warning' | 'info';

/**
 * Get toast icon name
 */
export const getToastIcon = (type: ToastType): string => {
  switch (type) {
    case 'success':
      return 'checkmark-circle';
    case 'error':
      return 'close-circle';
    case 'warning':
      return 'warning';
    case 'info':
      return 'information-circle';
    default:
      return 'information-circle';
  }
};

/**
 * Get toast color
 */
export const getToastColor = (type: ToastType): string => {
  switch (type) {
    case 'success':
      return '#10b981';
    case 'error':
      return '#ef4444';
    case 'warning':
      return '#f59e0b';
    case 'info':
      return '#3b82f6';
    default:
      return '#3b82f6';
  }
};

/**
 * Get default toast duration
 */
export const getDefaultToastDuration = (type: ToastType): number => {
  switch (type) {
    case 'error':
      return 5000; // Errors stay longer
    case 'warning':
      return 4000;
    case 'success':
      return 3000;
    case 'info':
      return 3000;
    default:
      return 3000;
  }
};

/**
 * Generate unique toast ID
 */
export const generateToastId = (): string => {
  return `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

