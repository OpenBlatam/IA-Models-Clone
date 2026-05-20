'use client';

import { useState } from 'react';
import { DashboardHeader } from '@/components/dashboard/header';
import GoogleCalendar from '@/components/calendar/google-calendar';
import { Button } from '@/components/ui/button';
import { Calendar, Plus, List, Grid, Clock } from 'lucide-react';

export default function CalendarPage() {
  const [view, setView] = useState<'week' | 'month' | 'day' | 'list'>('week');

  return (
    <div className="flex flex-col gap-4 p-4 md:p-8">
      <DashboardHeader
        heading="Calendario"
        text="Gestiona tus eventos y tareas"
      />
      
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button
              variant={view === 'week' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setView('week')}
            >
              <Clock className="mr-2 h-4 w-4" />
              Semana
            </Button>
            <Button
              variant={view === 'month' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setView('month')}
            >
              <Calendar className="mr-2 h-4 w-4" />
              Mes
            </Button>
            <Button
              variant={view === 'day' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setView('day')}
            >
              <Grid className="mr-2 h-4 w-4" />
              Día
            </Button>
            <Button
              variant={view === 'list' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setView('list')}
            >
              <List className="mr-2 h-4 w-4" />
              Lista
            </Button>
          </div>
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            Nuevo Evento
          </Button>
        </div>

        <div className="rounded-lg border bg-card">
          <GoogleCalendar
            calendarId="primary"
            height="800px"
          />
        </div>
      </div>
    </div>
  );
} 