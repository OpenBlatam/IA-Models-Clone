import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Re-export all utilities for backward compatibility
export * from './utils/date';
export * from './utils/labels';

export const cn = (...inputs: ClassValue[]): string => {
  return twMerge(clsx(inputs));
};

// Legacy function for backward compatibility
export const getProtocolPriorityColor = (priority: string): string => {
  const colors: Record<string, string> = {
    critical: 'text-red-600 bg-red-50',
    high: 'text-orange-600 bg-orange-50',
    medium: 'text-yellow-600 bg-yellow-50',
    low: 'text-blue-600 bg-blue-50',
  };
  return colors[priority] || 'text-gray-600 bg-gray-50';
};

