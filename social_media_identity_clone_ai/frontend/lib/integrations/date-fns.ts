import {
  format,
  formatDistance,
  formatDistanceToNow,
  isToday,
  isYesterday,
  isThisWeek,
  isThisMonth,
  isThisYear,
  parseISO,
  differenceInDays,
  differenceInHours,
  differenceInMinutes,
  addDays,
  addHours,
  addMinutes,
  startOfDay,
  endOfDay,
  startOfWeek,
  endOfWeek,
  startOfMonth,
  endOfMonth,
} from 'date-fns';

export const dateUtils = {
  format,
  formatDistance,
  formatDistanceToNow,
  isToday,
  isYesterday,
  isThisWeek,
  isThisMonth,
  isThisYear,
  parseISO,
  differenceInDays,
  differenceInHours,
  differenceInMinutes,
  addDays,
  addHours,
  addMinutes,
  startOfDay,
  endOfDay,
  startOfWeek,
  endOfWeek,
  startOfMonth,
  endOfMonth,
};

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
  return formatDistanceToNow(dateObj, { addSuffix: true });
};



