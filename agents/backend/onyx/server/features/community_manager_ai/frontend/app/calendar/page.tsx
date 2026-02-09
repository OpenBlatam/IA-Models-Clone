'use client';

export const dynamic = 'force-dynamic';

import { useEffect, useState } from 'react';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent } from '@/components/ui/Card';
import { Loading } from '@/components/ui/Loading';
import { Alert } from '@/components/ui/Alert';
import { calendarApi } from '@/lib/api';
import { CalendarEvent } from '@/types';
import { formatDate, getPlatformIcon } from '@/lib/utils';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { format, startOfWeek, endOfWeek, eachDayOfInterval, isSameDay, addWeeks, subWeeks } from 'date-fns';

export default function CalendarPage() {
  const [currentWeek, setCurrentWeek] = useState(new Date());
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchEvents();
  }, [currentWeek]);

  const fetchEvents = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const weekStart = startOfWeek(currentWeek);
      const weekEnd = endOfWeek(currentWeek);
      const data = await calendarApi.getEvents(
        weekStart.toISOString(),
        weekEnd.toISOString()
      );
      setEvents(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Error al cargar los eventos';
      setError(errorMessage);
      console.error('Error fetching calendar events:', err);
    } finally {
      setLoading(false);
    }
  };

  const weekDays = eachDayOfInterval({
    start: startOfWeek(currentWeek),
    end: endOfWeek(currentWeek),
  });

  const getEventsForDay = (day: Date) => {
    return events.filter((event) =>
      isSameDay(new Date(event.scheduled_time), day)
    );
  };

  const handlePreviousWeek = () => {
    setCurrentWeek(subWeeks(currentWeek, 1));
  };

  const handleNextWeek = () => {
    setCurrentWeek(addWeeks(currentWeek, 1));
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <Loading size="lg" text="Cargando calendario..." />
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Calendario</h1>
            <p className="mt-2 text-gray-600">Vista semanal de publicaciones</p>
          </div>
          <Alert variant="error" title="Error al cargar eventos">
            {error}
          </Alert>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Calendario</h1>
            <p className="mt-2 text-gray-600">Vista semanal de publicaciones</p>
          </div>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={handlePreviousWeek}
              className="rounded-lg p-2 text-gray-600 hover:bg-gray-100"
              aria-label="Semana anterior"
              tabIndex={0}
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <span className="text-lg font-medium">
              {format(startOfWeek(currentWeek), 'd MMM')} -{' '}
              {format(endOfWeek(currentWeek), 'd MMM yyyy')}
            </span>
            <button
              type="button"
              onClick={handleNextWeek}
              className="rounded-lg p-2 text-gray-600 hover:bg-gray-100"
              aria-label="Semana siguiente"
              tabIndex={0}
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          </div>
        </div>

        <div className="grid grid-cols-7 gap-4">
          {weekDays.map((day: Date, index: number) => {
            const dayEvents = getEventsForDay(day);
            const isToday = isSameDay(day, new Date());

            return (
              <div key={index} className="flex flex-col">
                <div
                  className={`mb-2 text-center text-sm font-medium ${
                    isToday ? 'text-primary-600' : 'text-gray-700'
                  }`}
                >
                  {format(day, 'EEE')}
                </div>
                <div
                  className={`mb-1 text-center text-2xl font-bold ${
                    isToday ? 'text-primary-600' : 'text-gray-900'
                  }`}
                >
                  {format(day, 'd')}
                </div>
                <div className="flex-1 space-y-2 min-h-[200px]">
                  {dayEvents.map((event) => (
                    <Card key={event.id} className="p-2">
                      <CardContent className="p-0">
                        <p className="mb-1 text-xs font-medium text-gray-900 line-clamp-2">
                          {event.content}
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {event.platforms.map((platform) => (
                            <span
                              key={platform}
                              className="text-xs"
                              title={platform}
                            >
                              {getPlatformIcon(platform)}
                            </span>
                          ))}
                        </div>
                        <p className="mt-1 text-xs text-gray-500">
                          {format(new Date(event.scheduled_time), 'HH:mm')}
                        </p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Layout>
  );
}

