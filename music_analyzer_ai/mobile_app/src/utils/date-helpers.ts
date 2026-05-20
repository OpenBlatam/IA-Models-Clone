import { format, formatDistance, formatRelative, isToday, isYesterday } from 'date-fns';
import { enUS, es } from 'date-fns/locale';

const locales = { en: enUS, es };

/**
 * Format date with locale support
 */
export function formatDate(
  date: Date | string | number,
  formatStr = 'PPp',
  locale = 'en'
): string {
  const dateObj =
    typeof date === 'string' || typeof date === 'number'
      ? new Date(date)
      : date;

  return format(dateObj, formatStr, {
    locale: locales[locale as keyof typeof locales] || enUS,
  });
}

/**
 * Format relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(
  date: Date | string | number,
  locale = 'en'
): string {
  const dateObj =
    typeof date === 'string' || typeof date === 'number'
      ? new Date(date)
      : date;

  if (isToday(dateObj)) {
    return formatDistance(dateObj, new Date(), {
      addSuffix: true,
      locale: locales[locale as keyof typeof locales] || enUS,
    });
  }

  if (isYesterday(dateObj)) {
    return locale === 'es' ? 'Ayer' : 'Yesterday';
  }

  return formatRelative(dateObj, new Date(), {
    locale: locales[locale as keyof typeof locales] || enUS,
  });
}

/**
 * Format duration in milliseconds to readable format
 */
export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}:${String(minutes % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  }

  return `${minutes}:${String(seconds % 60).padStart(2, '0')}`;
}

/**
 * Get time ago string
 */
export function getTimeAgo(date: Date | string | number): string {
  return formatRelativeTime(date);
}

/**
 * Check if date is recent (within last N days)
 */
export function isRecent(date: Date | string | number, days = 7): boolean {
  const dateObj =
    typeof date === 'string' || typeof date === 'number'
      ? new Date(date)
      : date;
  const now = new Date();
  const diffTime = Math.abs(now.getTime() - dateObj.getTime());
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return diffDays <= days;
}

