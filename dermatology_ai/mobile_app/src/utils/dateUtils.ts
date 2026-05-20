import { format, formatDistance, formatRelative, isToday, isYesterday, parseISO } from 'date-fns';
import { es, enUS } from 'date-fns/locale';

/**
 * Date utilities with localization support
 */

const locales = {
  es,
  en: enUS,
};

export const formatDate = (
  date: Date | string,
  formatStr: string = 'PP',
  locale: 'es' | 'en' = 'es'
): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, formatStr, { locale: locales[locale] });
};

export const formatRelativeTime = (
  date: Date | string,
  locale: 'es' | 'en' = 'es'
): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return formatRelative(dateObj, new Date(), { locale: locales[locale] });
};

export const formatDistanceTime = (
  date: Date | string,
  locale: 'es' | 'en' = 'es'
): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return formatDistance(dateObj, new Date(), {
    addSuffix: true,
    locale: locales[locale],
  });
};

export const getSmartDate = (
  date: Date | string,
  locale: 'es' | 'en' = 'es'
): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  
  if (isToday(dateObj)) {
    return locale === 'es' ? 'Hoy' : 'Today';
  }
  
  if (isYesterday(dateObj)) {
    return locale === 'es' ? 'Ayer' : 'Yesterday';
  }
  
  return formatRelativeTime(dateObj, locale);
};

export const getTimeAgo = (
  date: Date | string,
  locale: 'es' | 'en' = 'es'
): string => {
  return formatDistanceTime(date, locale);
};

