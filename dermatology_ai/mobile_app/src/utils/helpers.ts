import { format } from 'date-fns';

/**
 * Format a date to a readable string
 */
export const formatDate = (date: Date | string): string => {
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    return format(dateObj, "d MMM yyyy 'a las' HH:mm");
  } catch (error) {
    return 'Fecha no disponible';
  }
};

/**
 * Get score color based on value
 */
export const getScoreColor = (score: number): string => {
  if (score >= 80) return '#10b981'; // Green
  if (score >= 60) return '#f59e0b'; // Yellow
  return '#ef4444'; // Red
};

/**
 * Get score label based on value
 */
export const getScoreLabel = (score: number): string => {
  if (score >= 80) return 'Excelente';
  if (score >= 60) return 'Bueno';
  if (score >= 40) return 'Regular';
  return 'Necesita Mejora';
};

/**
 * Validate image URI
 */
export const isValidImageUri = (uri: string | undefined): boolean => {
  if (!uri) return false;
  return uri.startsWith('http') || uri.startsWith('file://') || uri.startsWith('content://');
};

/**
 * Truncate text
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

/**
 * Capitalize first letter
 */
export const capitalize = (str: string): string => {
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

