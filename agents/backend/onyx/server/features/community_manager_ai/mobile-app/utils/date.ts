import { format, formatDistanceToNow, isToday, isYesterday, isThisWeek, isThisYear } from 'date-fns';

/**
 * Date formatting utilities
 */

export function formatDate(date: Date | string, formatStr: string = 'MMM dd, yyyy'): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return format(dateObj, formatStr);
}

export function formatDateTime(date: Date | string): string {
  return formatDate(date, 'MMM dd, yyyy HH:mm');
}

export function formatTime(date: Date | string): string {
  return formatDate(date, 'HH:mm');
}

export function formatRelative(date: Date | string): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return formatDistanceToNow(dateObj, { addSuffix: true });
}

export function formatSmart(date: Date | string): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;

  if (isToday(dateObj)) {
    return `Today, ${formatTime(dateObj)}`;
  }

  if (isYesterday(dateObj)) {
    return `Yesterday, ${formatTime(dateObj)}`;
  }

  if (isThisWeek(dateObj)) {
    return format(dateObj, 'EEEE, HH:mm');
  }

  if (isThisYear(dateObj)) {
    return format(dateObj, 'MMM dd, HH:mm');
  }

  return formatDateTime(dateObj);
}

export function getTimeAgo(date: Date | string): string {
  return formatRelative(date);
}

export function isDateValid(date: Date | string): boolean {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return !isNaN(dateObj.getTime());
}

export function addDays(date: Date | string, days: number): Date {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const result = new Date(dateObj);
  result.setDate(result.getDate() + days);
  return result;
}

export function addHours(date: Date | string, hours: number): Date {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const result = new Date(dateObj);
  result.setHours(result.getHours() + hours);
  return result;
}

export function getStartOfDay(date: Date | string): Date {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const result = new Date(dateObj);
  result.setHours(0, 0, 0, 0);
  return result;
}

export function getEndOfDay(date: Date | string): Date {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const result = new Date(dateObj);
  result.setHours(23, 59, 59, 999);
  return result;
}

