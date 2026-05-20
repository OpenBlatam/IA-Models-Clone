/**
 * Calendar API
 * Handles all calendar-related API operations
 */

import { apiGet } from './client';
import { API_ENDPOINTS } from '@/lib/config/constants';
import type { CalendarEvent } from '@/types';

/**
 * Get calendar events within a date range
 * @param startDate - Optional start date (ISO string)
 * @param endDate - Optional end date (ISO string)
 * @returns Array of calendar events
 */
export const getCalendarEvents = async (
  startDate?: string,
  endDate?: string
): Promise<CalendarEvent[]> => {
  return apiGet<CalendarEvent[]>(API_ENDPOINTS.CALENDAR, {
    params: {
      start_date: startDate,
      end_date: endDate,
    },
  });
};

/**
 * Get daily calendar events for a specific date
 * @param date - The date (ISO string)
 * @returns Array of calendar events for the day
 */
export const getDailyEvents = async (date: string): Promise<CalendarEvent[]> => {
  return apiGet<CalendarEvent[]>(`${API_ENDPOINTS.CALENDAR}/daily`, {
    params: { date },
  });
};

/**
 * Get weekly calendar events
 * @param startDate - Optional start date (ISO string)
 * @returns Array of calendar events for the week
 */
export const getWeeklyEvents = async (startDate?: string): Promise<CalendarEvent[]> => {
  return apiGet<CalendarEvent[]>(`${API_ENDPOINTS.CALENDAR}/weekly`, {
    params: startDate ? { start_date: startDate } : undefined,
  });
};


