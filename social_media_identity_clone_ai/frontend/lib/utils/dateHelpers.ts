import { format, formatDistance, formatRelative, isToday, isYesterday, isTomorrow, parseISO } from 'date-fns';

export const formatDateShort = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, 'MMM d, yyyy');
};

export const formatDateLong = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, 'MMMM d, yyyy');
};

export const formatDateTime = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, 'MMM d, yyyy h:mm a');
};

export const formatTimeAgo = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return formatDistance(dateObj, new Date(), { addSuffix: true });
};

export const formatRelativeTime = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return formatRelative(dateObj, new Date());
};

export const formatSmartDate = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;

  if (isToday(dateObj)) {
    return `Today at ${format(dateObj, 'h:mm a')}`;
  }

  if (isYesterday(dateObj)) {
    return `Yesterday at ${format(dateObj, 'h:mm a')}`;
  }

  if (isTomorrow(dateObj)) {
    return `Tomorrow at ${format(dateObj, 'h:mm a')}`;
  }

  return formatDateTime(dateObj);
};

export const getDaysBetween = (startDate: Date | string, endDate: Date | string): number => {
  const start = typeof startDate === 'string' ? parseISO(startDate) : startDate;
  const end = typeof endDate === 'string' ? parseISO(endDate) : endDate;
  const diffTime = Math.abs(end.getTime() - start.getTime());
  return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
};

export const isDateInRange = (date: Date | string, startDate: Date | string, endDate: Date | string): boolean => {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  const start = typeof startDate === 'string' ? parseISO(startDate) : startDate;
  const end = typeof endDate === 'string' ? parseISO(endDate) : endDate;
  return dateObj >= start && dateObj <= end;
};



