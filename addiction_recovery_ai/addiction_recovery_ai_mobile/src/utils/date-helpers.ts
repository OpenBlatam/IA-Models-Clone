import { format, parseISO, differenceInDays, differenceInHours, isToday, isYesterday } from 'date-fns';

// Pure functions for date operations

export function formatDate(date: Date | string, formatStr: string = 'PP'): string {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, formatStr);
}

export function formatRelativeTime(date: Date | string): string {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  const now = new Date();

  if (isToday(dateObj)) {
    return 'Hoy';
  }

  if (isYesterday(dateObj)) {
    return 'Ayer';
  }

  const daysDiff = differenceInDays(now, dateObj);
  if (daysDiff < 7) {
    return `Hace ${daysDiff} días`;
  }

  const hoursDiff = differenceInHours(now, dateObj);
  if (hoursDiff < 24) {
    return `Hace ${hoursDiff} horas`;
  }

  return formatDate(dateObj);
}

export function getDaysSince(date: Date | string): number {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return differenceInDays(new Date(), dateObj);
}

export function formatTime(date: Date | string): string {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, 'HH:mm');
}

export function formatDateTime(date: Date | string): string {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, 'PPp');
}

