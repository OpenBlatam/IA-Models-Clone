/**
 * Date utility functions.
 * Provides helper functions for date manipulation and formatting.
 */

import { format, formatDistance, formatRelative, isToday, isYesterday, isTomorrow, parseISO } from 'date-fns';
import { es } from 'date-fns/locale';

/**
 * Formats a date to a readable string.
 * @param date - Date to format
 * @param formatStr - Format string (default: 'PP')
 * @returns Formatted date string
 */
export function formatDate(
  date: Date | string | number,
  formatStr: string = 'PP'
): string {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? parseISO(String(date))
    : date;

  if (isNaN(dateObj.getTime())) {
    return 'Fecha inválida';
  }

  return format(dateObj, formatStr, { locale: es });
}

/**
 * Formats a date as relative time (e.g., "hace 2 horas").
 * @param date - Date to format
 * @returns Relative time string
 */
export function formatRelativeDate(
  date: Date | string | number
): string {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? parseISO(String(date))
    : date;

  if (isNaN(dateObj.getTime())) {
    return 'Fecha inválida';
  }

  return formatDistance(dateObj, new Date(), {
    addSuffix: true,
    locale: es,
  });
}

/**
 * Formats a date with smart relative formatting.
 * Shows "hoy", "ayer", "mañana" for recent dates.
 * @param date - Date to format
 * @returns Smart formatted date string
 */
export function formatSmartDate(
  date: Date | string | number
): string {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? parseISO(String(date))
    : date;

  if (isNaN(dateObj.getTime())) {
    return 'Fecha inválida';
  }

  if (isToday(dateObj)) {
    return `Hoy, ${format(dateObj, 'HH:mm')}`;
  }

  if (isYesterday(dateObj)) {
    return `Ayer, ${format(dateObj, 'HH:mm')}`;
  }

  if (isTomorrow(dateObj)) {
    return `Mañana, ${format(dateObj, 'HH:mm')}`;
  }

  return formatDate(dateObj);
}

/**
 * Gets the start of a day.
 * @param date - Date
 * @returns Start of day
 */
export function startOfDay(date: Date = new Date()): Date {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  return d;
}

/**
 * Gets the end of a day.
 * @param date - Date
 * @returns End of day
 */
export function endOfDay(date: Date = new Date()): Date {
  const d = new Date(date);
  d.setHours(23, 59, 59, 999);
  return d;
}

/**
 * Checks if a date is in the past.
 * @param date - Date to check
 * @returns True if date is in the past
 */
export function isPast(date: Date | string | number): boolean {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? parseISO(String(date))
    : date;

  return dateObj.getTime() < Date.now();
}

/**
 * Checks if a date is in the future.
 * @param date - Date to check
 * @returns True if date is in the future
 */
export function isFuture(date: Date | string | number): boolean {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? parseISO(String(date))
    : date;

  return dateObj.getTime() > Date.now();
}

/**
 * Adds days to a date.
 * @param date - Base date
 * @param days - Number of days to add
 * @returns New date
 */
export function addDays(date: Date, days: number): Date {
  const result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
}

/**
 * Gets the difference in days between two dates.
 * @param date1 - First date
 * @param date2 - Second date
 * @returns Difference in days
 */
export function differenceInDays(
  date1: Date | string | number,
  date2: Date | string | number = new Date()
): number {
  const d1 = typeof date1 === 'string' || typeof date1 === 'number'
    ? parseISO(String(date1))
    : date1;
  const d2 = typeof date2 === 'string' || typeof date2 === 'number'
    ? parseISO(String(date2))
    : date2;

  const diffTime = Math.abs(d2.getTime() - d1.getTime());
  return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
}

