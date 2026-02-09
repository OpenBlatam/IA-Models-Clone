import { format, formatDistance, formatRelative, isToday, isYesterday } from 'date-fns';
import { enUS, es } from 'date-fns/locale';

const locales = { en: enUS, es };

export function formatDate(
  date: Date | string,
  formatStr = 'PPp',
  locale = 'en'
): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return format(dateObj, formatStr, {
    locale: locales[locale as keyof typeof locales] || enUS,
  });
}

export function formatRelativeTime(
  date: Date | string,
  locale = 'en'
): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  
  if (isToday(dateObj)) {
    return formatDistance(dateObj, new Date(), {
      addSuffix: true,
      locale: locales[locale as keyof typeof locales] || enUS,
    });
  }
  
  if (isYesterday(dateObj)) {
    return 'Yesterday';
  }
  
  return formatRelative(dateObj, new Date(), {
    locale: locales[locale as keyof typeof locales] || enUS,
  });
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

