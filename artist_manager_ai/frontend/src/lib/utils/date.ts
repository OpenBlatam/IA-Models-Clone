import { format, parseISO, isValid, addDays, subDays, startOfDay, endOfDay } from 'date-fns';
import { es } from 'date-fns/locale';

export const formatDate = (date: string | Date): string => {
  const d = typeof date === 'string' ? parseISO(date) : date;
  if (!isValid(d)) return '';
  return format(d, 'dd/MM/yyyy', { locale: es });
};

export const formatTime = (date: string | Date): string => {
  const d = typeof date === 'string' ? parseISO(date) : date;
  if (!isValid(d)) return '';
  return format(d, 'HH:mm', { locale: es });
};

export const formatDateTime = (date: string | Date): string => {
  const d = typeof date === 'string' ? parseISO(date) : date;
  if (!isValid(d)) return '';
  return format(d, "dd/MM/yyyy 'a las' HH:mm", { locale: es });
};

export const formatDateLong = (date: string | Date): string => {
  const d = typeof date === 'string' ? parseISO(date) : date;
  if (!isValid(d)) return '';
  return format(d, "EEEE, d 'de' MMMM 'de' yyyy", { locale: es });
};

export const getTimeFromString = (timeString: string): Date => {
  const [hours, minutes] = timeString.split(':').map(Number);
  const date = new Date();
  date.setHours(hours, minutes || 0, 0, 0);
  return date;
};

export const getDaysOfWeekNames = (days: number[]): string[] => {
  const dayNames = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'];
  return days.map((day) => dayNames[day] || '');
};

export const getDateRange = (days: number) => {
  const today = new Date();
  return {
    start: startOfDay(subDays(today, days)),
    end: endOfDay(today),
  };
};

export const isToday = (date: string | Date): boolean => {
  const d = typeof date === 'string' ? parseISO(date) : date;
  if (!isValid(d)) return false;
  const today = new Date();
  return format(d, 'yyyy-MM-dd') === format(today, 'yyyy-MM-dd');
};

export const isPast = (date: string | Date): boolean => {
  const d = typeof date === 'string' ? parseISO(date) : date;
  if (!isValid(d)) return false;
  return d < new Date();
};

export const isFuture = (date: string | Date): boolean => {
  const d = typeof date === 'string' ? parseISO(date) : date;
  if (!isValid(d)) return false;
  return d > new Date();
};

