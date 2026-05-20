'use client';

import { useEffect, useState, useRef } from 'react';
import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import interactionPlugin from '@fullcalendar/interaction';
import listPlugin from '@fullcalendar/list';
import type { EventInput } from '@fullcalendar/core';
import { EventModal } from './event-modal';
import { AINotesModal } from './ai-notes-modal';
import { Button } from '@/components/ui/button';
import { Plus, Calendar as CalendarIcon, List, Grid, Clock, Sparkles, MapPin, CheckCircle, XCircle, Link, Link2Off } from 'lucide-react';
import { cn } from '@/lib/utils';
import { addWeeks, setHours, setMinutes, setSeconds, nextSaturday, startOfWeek, endOfWeek, format, subWeeks } from 'date-fns';
import * as Tooltip from '@radix-ui/react-tooltip';
import clsx from 'clsx';
import { motion } from 'framer-motion';
import { Popover, PopoverTrigger, PopoverContent } from '@radix-ui/react-popover';
import { ScrollArea } from '@radix-ui/react-scroll-area';
import { CalendarBlank, VideoCamera, UsersThree, Clock as ClockPhosphor } from 'phosphor-react';
import esLocale from '@fullcalendar/core/locales/es';
import { es } from 'date-fns/locale';
import toast from 'react-hot-toast';
import Confetti from 'react-confetti';
import { DayPicker } from 'react-day-picker';
import 'react-day-picker/dist/style.css';
import PerfectScrollbar from 'react-perfect-scrollbar';
import 'react-perfect-scrollbar/dist/css/styles.css';
import { useSession } from 'next-auth/react';

interface GoogleCalendarProps {
  calendarId?: string;
  height?: string;
}

const ZOOM_LINK = 'https://us04web.zoom.us/j/1234567890?pwd=abcdefg';

export default function GoogleCalendar({ calendarId = 'primary', height = '800px' }: GoogleCalendarProps) {
  const { data: session } = useSession();
  const [events, setEvents] = useState<EventInput[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isAINotesModalOpen, setIsAINotesModalOpen] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<any>(null);
  const [view, setView] = useState<'week' | 'month' | 'day' | 'list'>('week');
  const calendarRef = useRef<any>(null);
  const calendarContainerRef = useRef<HTMLDivElement>(null);
  const [currentDate, setCurrentDate] = useState(new Date());
  const [showConfetti, setShowConfetti] = useState(false);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [isGoogleCalendarConnected, setIsGoogleCalendarConnected] = useState(false);

  useEffect(() => {
    const loadEvents = async () => {
      try {
        if (isGoogleCalendarConnected && session?.accessToken) {
          const response = await fetch(
            `https://www.googleapis.com/calendar/v3/calendars/${calendarId}/events?key=${process.env.NEXT_PUBLIC_GOOGLE_CALENDAR_API_KEY}`,
            {
              headers: {
                Authorization: `Bearer ${session.accessToken}`,
              },
            }
          );
          const data = await response.json();
          const formattedEvents = Array.isArray(data.items)
            ? data.items.map((event: any) => ({
                id: event.id,
                title: event.summary,
                start: event.start.dateTime || event.start.date,
                end: event.end.dateTime || event.end.date,
                description: event.description,
                location: event.location,
                backgroundColor: event.colorId ? getColorFromId(event.colorId) : '#3788d8',
                borderColor: event.colorId ? getColorFromId(event.colorId) : '#3788d8',
                textColor: '#ffffff',
                extendedProps: {
                  description: event.description,
                  location: event.location,
                  attendees: event.attendees,
                },
              }))
            : [];

          // Agregar eventos de clase vía Zoom cada sábado de 6 a 8 pm durante 4 semanas
          const now = new Date();
          let firstSaturday = nextSaturday(now);
          firstSaturday = setHours(setMinutes(setSeconds(firstSaturday, 0), 0), 18); // 6:00pm
          const zoomEvents = Array.from({ length: 4 }).map((_, i) => {
            const start = addWeeks(firstSaturday, i);
            const end = setHours(setMinutes(setSeconds(addWeeks(firstSaturday, i), 0), 0), 20); // 8:00pm
            return {
              id: `zoom-class-${i}`,
              title: 'Clase vía Zoom',
              start,
              end,
              description: `Clase recurrente de Zoom.\nEnlace: ${ZOOM_LINK}`,
              location: 'Zoom',
              backgroundColor: '#039be5',
              borderColor: '#039be5',
              textColor: '#fff',
              extendedProps: {
                description: 'Clase recurrente de Zoom',
                location: 'Zoom',
                zoomLink: ZOOM_LINK,
              },
            };
          });

          setEvents([...formattedEvents, ...zoomEvents]);
        }
      } catch (error) {
        toast.error('Error al cargar los eventos de Google Calendar');
      }
    };

    loadEvents();
  }, [calendarId, isGoogleCalendarConnected, session?.accessToken]);

  // Atajo de teclado Shift+W para vista semana
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      if (["input", "textarea", "select"].includes(tag)) return;

      if (e.shiftKey && (e.key === 'w' || e.key === 'W')) {
        handleChangeView('week');
      }
      if (e.key === '1') {
        handleChangeView('day');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Auto-scroll horizontal al acercar el mouse al borde derecho
  useEffect(() => {
    const container = calendarContainerRef.current;
    if (!container) return;

    let animationFrame: number | null = null;
    let isScrolling = false;
    const edgeThreshold = 60; // px
    const scrollSpeed = 10; // px por frame

    const handleMouseMove = (e: MouseEvent) => {
      const { left, right } = container.getBoundingClientRect();
      const mouseX = e.clientX;

      if (right - mouseX < edgeThreshold) {
        isScrolling = true;
        scrollRight();
      } else {
        isScrolling = false;
      }
    };

    const scrollRight = () => {
      if (!isScrolling) return;
      container.scrollLeft += scrollSpeed;
      animationFrame = requestAnimationFrame(scrollRight);
    };

    container.addEventListener('mousemove', handleMouseMove);
    container.addEventListener('mouseleave', () => { isScrolling = false; });

    return () => {
      container.removeEventListener('mousemove', handleMouseMove);
      container.removeEventListener('mouseleave', () => { isScrolling = false; });
      if (animationFrame) cancelAnimationFrame(animationFrame);
    };
  }, []);

  const getColorFromId = (colorId: string) => {
    const colors: { [key: string]: string } = {
      '1': '#7986cb', // Lavender
      '2': '#33b679', // Sage
      '3': '#8e24aa', // Grape
      '4': '#e67c73', // Flamingo
      '5': '#f6c026', // Banana
      '6': '#f5511d', // Tangerine
      '7': '#039be5', // Peacock
      '8': '#616161', // Graphite
      '9': '#3f51b5', // Blueberry
      '10': '#0b8043', // Basil
      '11': '#d60000', // Tomato
    };
    return colors[colorId] || '#3788d8';
  };

  const handleEventSave = (eventData: any) => {
    const newEvent = {
      id: eventData.id,
      title: eventData.title,
      start: eventData.start,
      end: eventData.end,
      description: eventData.description,
      location: eventData.location,
      backgroundColor: eventData.color || '#3788d8',
      borderColor: eventData.color || '#3788d8',
      textColor: '#ffffff',
      extendedProps: {
        description: eventData.description,
        location: eventData.location,
      },
    };
    setEvents(prevEvents => {
      const exists = prevEvents.some(e => e.id === newEvent.id);
      return exists
        ? prevEvents.map(e => e.id === newEvent.id ? newEvent : e)
        : [...prevEvents, newEvent];
    });
    toast.custom((t) => (
      <div className={`flex items-center gap-3 px-4 py-3 rounded-xl shadow-lg bg-white/90 dark:bg-muted/90 border border-green-200 animate-in fade-in ${t.visible ? 'opacity-100' : 'opacity-0'}`}
           style={{ minWidth: 220 }}>
        <CheckCircle size={28} className="text-green-500" />
        <span className="font-semibold text-green-700 dark:text-green-400">¡Evento creado exitosamente!</span>
      </div>
    ), { duration: 3500 });
    setShowConfetti(true);
    setTimeout(() => setShowConfetti(false), 2500);
  };

  const handleChangeView = (newView: 'week' | 'month' | 'day' | 'list') => {
    setView(newView);
    if (calendarRef.current) {
      const api = calendarRef.current.getApi();
      if (newView === 'week') api.changeView('timeGridWeek');
      if (newView === 'month') api.changeView('dayGridMonth');
      if (newView === 'day') api.changeView('timeGridDay');
      if (newView === 'list') api.changeView('listWeek');
    }
  };

  // Cambiar semana
  const goToPrevWeek = () => {
    const newDate = subWeeks(currentDate, 1);
    setCurrentDate(newDate);
    if (calendarRef.current) {
      calendarRef.current.getApi().gotoDate(newDate);
    }
  };
  const goToNextWeek = () => {
    const newDate = addWeeks(currentDate, 1);
    setCurrentDate(newDate);
    if (calendarRef.current) {
      calendarRef.current.getApi().gotoDate(newDate);
    }
  };
  const goToToday = () => {
    setCurrentDate(new Date());
    if (calendarRef.current) {
      calendarRef.current.getApi().today();
    }
  };

  // Rango de la semana actual
  const weekStart = startOfWeek(currentDate, { weekStartsOn: 1, locale: es });
  const weekEnd = endOfWeek(currentDate, { weekStartsOn: 1, locale: es });
  const weekLabel = `${format(weekStart, "d 'de' MMMM", { locale: es })} - ${format(weekEnd, "d 'de' MMMM yyyy", { locale: es })}`;

  const handleGoogleCalendarConnect = async () => {
    if (!isGoogleCalendarConnected) {
      try {
        const response = await fetch('/api/calendar/connect', {
          method: 'POST',
        });
        if (response.ok) {
          setIsGoogleCalendarConnected(true);
          toast.success('Calendario de Google conectado exitosamente');
        } else {
          toast.error('Error al conectar con Google Calendar');
        }
      } catch (error) {
        toast.error('Error al conectar con Google Calendar');
      }
    } else {
      try {
        const response = await fetch('/api/calendar/disconnect', {
          method: 'POST',
        });
        if (response.ok) {
          setIsGoogleCalendarConnected(false);
          toast.success('Calendario de Google desconectado exitosamente');
        } else {
          toast.error('Error al desconectar de Google Calendar');
        }
      } catch (error) {
        toast.error('Error al desconectar de Google Calendar');
      }
    }
  };

  return (
    <motion.div
      className="flex flex-col gap-6"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      {/* Cabecera glassmorphism animada */}
      <motion.div
        className="sticky top-0 z-20 flex flex-col gap-2 bg-white/70 dark:bg-muted/80 px-6 pt-6 pb-2 rounded-b-2xl shadow-2xl border-b border-border backdrop-blur-xl animate-in fade-in"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-2">
          <h1 className="text-3xl font-bold text-primary tracking-tight flex items-center gap-2">
            <CalendarBlank size={32} className="text-violet-500" /> Calendario
          </h1>
          <div className="flex items-center gap-2 ml-auto">
            <button onClick={goToPrevWeek} className="btn btn-circle btn-ghost hover:bg-accent transition-colors">
              <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M15 19l-7-7 7-7"/></svg>
            </button>
            <span className="font-semibold text-primary bg-white/80 dark:bg-muted/80 px-4 py-2 rounded-xl border border-border shadow-sm text-sm animate-in fade-in">
              {weekLabel}
            </span>
            <button onClick={goToToday} className="btn btn-primary px-4 py-2 font-semibold shadow-md">Hoy</button>
            <button onClick={goToNextWeek} className="btn btn-circle btn-ghost hover:bg-accent transition-colors">
              <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M9 5l7 7-7 7"/></svg>
            </button>
            <motion.button
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => {
                setSelectedEvent(null);
                setIsModalOpen(true);
              }}
              className="ml-2 flex items-center gap-2 px-5 py-2 rounded-xl font-semibold bg-gradient-to-r from-violet-500 to-blue-500 text-white shadow-lg hover:from-violet-600 hover:to-blue-600 transition-all focus:outline-none focus:ring-2 focus:ring-violet-400 animate-in fade-in"
            >
              <Plus className="h-5 w-5" /> Nuevo Evento
            </motion.button>
            <button
              onClick={() => setShowDatePicker(v => !v)}
              className="ml-2 flex items-center gap-2 px-4 py-2 rounded-xl font-semibold bg-gradient-to-r from-blue-500 to-violet-500 text-white shadow hover:from-blue-600 hover:to-violet-600 transition-all"
            >
              <CalendarIcon className="h-5 w-5" /> Ir a fecha
            </button>
            <motion.button
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.97 }}
              onClick={handleGoogleCalendarConnect}
              className={cn(
                "ml-2 flex items-center gap-2 px-5 py-2 rounded-xl font-semibold shadow-lg transition-all focus:outline-none focus:ring-2 focus:ring-violet-400 animate-in fade-in",
                isGoogleCalendarConnected
                  ? "bg-gradient-to-r from-red-500 to-pink-500 text-white hover:from-red-600 hover:to-pink-600"
                  : "bg-gradient-to-r from-violet-500 to-blue-500 text-white hover:from-violet-600 hover:to-blue-600"
              )}
            >
              {isGoogleCalendarConnected ? (
                <>
                  <Link2Off className="h-5 w-5" /> Desconectar Google Calendar
                </>
              ) : (
                <>
                  <Link className="h-5 w-5" /> Conectar Google Calendar
                </>
              )}
            </motion.button>
          </div>
        </div>
        {/* Tabs animados */}
        <motion.div layout className="flex gap-2">
          {[
            { key: 'week', label: 'Semana', icon: <ClockPhosphor size={20} className="mr-2" />, shortcut: 'Shift + W' },
            { key: 'month', label: 'Mes', icon: <CalendarIcon className="mr-2 h-5 w-5" /> },
            { key: 'day', label: 'Día', icon: <Grid className="mr-2 h-5 w-5" />, shortcut: '1' },
            { key: 'list', label: 'Lista', icon: <List className="mr-2 h-5 w-5" /> },
          ].map(tab => (
            <motion.button
              key={tab.key}
              onClick={() => handleChangeView(tab.key as any)}
              className={clsx(
                'flex items-center gap-1 px-5 py-2 rounded-xl font-medium transition-all',
                view === tab.key
                  ? 'bg-gradient-to-r from-violet-500 to-blue-500 text-white shadow-lg scale-105'
                  : 'bg-white/80 dark:bg-muted border border-border text-primary hover:bg-accent/60'
              )}
              layout
              whileTap={{ scale: 0.97 }}
            >
              {tab.icon}
              {tab.label}
              {tab.shortcut && (
                <span className="ml-2 px-2 py-0.5 rounded bg-violet-100 text-violet-700 text-xs font-mono border border-violet-200 animate-in fade-in">
                  {tab.shortcut}
                </span>
              )}
            </motion.button>
          ))}
        </motion.div>
      </motion.div>

      {/* Calendario con glassmorphism y eventos animados */}
      <PerfectScrollbar style={{ borderRadius: '1rem' }}>
        <motion.div
          ref={calendarContainerRef}
          className="calendar-container rounded-2xl border bg-white/80 dark:bg-muted/80 shadow-2xl p-2 backdrop-blur-xl animate-in fade-in overflow-x-auto"
          style={{ height }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
        >
          <FullCalendar
            ref={calendarRef}
            plugins={[dayGridPlugin, timeGridPlugin, interactionPlugin, listPlugin]}
            initialView={view === 'week' ? 'timeGridWeek' : 
                        view === 'month' ? 'dayGridMonth' : 
                        view === 'day' ? 'timeGridDay' : 'listWeek'}
            headerToolbar={false}
            eventContent={renderEventContent}
            locales={[esLocale]}
            locale="es"
            views={{
              timeGridWeek: {
                titleFormat: { year: 'numeric', month: 'long', day: 'numeric' },
                slotLabelFormat: { hour: '2-digit', minute: '2-digit', hour12: true },
                slotMinTime: '08:00:00',
                slotMaxTime: '20:00:00',
              },
              dayGridMonth: {
                titleFormat: { year: 'numeric', month: 'long' },
              },
              timeGridDay: {
                titleFormat: { year: 'numeric', month: 'long', day: 'numeric' },
                slotLabelFormat: { hour: '2-digit', minute: '2-digit', hour12: true },
              },
              listWeek: {
                titleFormat: { year: 'numeric', month: 'long', day: 'numeric' },
                listDayFormat: { weekday: 'long', month: 'long', day: 'numeric' },
                listDaySideFormat: { year: 'numeric', month: 'long', day: 'numeric' },
              }
            }}
            events={events}
            editable={true}
            selectable={true}
            selectMirror={true}
            dayMaxEvents={true}
            weekends={true}
            nowIndicator={true}
            eventTimeFormat={{
              hour: '2-digit',
              minute: '2-digit',
              meridiem: true,
              hour12: true
            }}
            eventDisplay="block"
            eventMinHeight={25}
            slotDuration="00:30:00"
            slotLabelInterval="01:00"
            allDaySlot={true}
            allDayText="Todo el día"
            eventClick={(info) => {
              setSelectedEvent(info.event);
              setIsModalOpen(true);
            }}
            eventDrop={(info) => {
            }}
            eventResize={(info) => {
            }}
            select={(info) => {
              setSelectedEvent(null);
              setIsModalOpen(true);
            }}
            height="100%"
            stickyHeaderDates={true}
            dayHeaderFormat={{ weekday: 'long' }}
            firstDay={1}
            businessHours={{
              daysOfWeek: [1, 2, 3, 4, 5],
              startTime: '09:00',
              endTime: '18:00',
            }}
            eventOverlap={false}
            eventConstraint="businessHours"
            eventDidMount={(info) => {
              info.el.title = info.event.extendedProps.description || info.event.title;
            }}
          />
        </motion.div>
      </PerfectScrollbar>

      {/* Confetti en la parte superior del layout */}
      {showConfetti && (
        <div className="fixed inset-0 z-[9999] pointer-events-none">
          <Confetti width={window.innerWidth} height={window.innerHeight} recycle={false} numberOfPieces={250} />
        </div>
      )}

      {/* Date Picker avanzado con glassmorphism y cierre al hacer click fuera */}
      {showDatePicker && (
        <div className="fixed inset-0 z-50 flex items-start justify-end bg-black/10" onClick={() => setShowDatePicker(false)}>
          <div
            className="mt-24 mr-8 bg-white/90 dark:bg-muted/90 rounded-2xl shadow-2xl p-4 animate-in fade-in backdrop-blur-xl"
            onClick={e => e.stopPropagation()}
          >
            <DayPicker
              mode="single"
              selected={currentDate}
              onSelect={date => {
                if (date) {
                  setCurrentDate(date);
                  if (calendarRef.current) calendarRef.current.getApi().gotoDate(date);
                  setShowDatePicker(false);
                }
              }}
              locale={es}
            />
          </div>
        </div>
      )}

      <EventModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        event={selectedEvent}
        onSave={handleEventSave}
      />

      <AINotesModal
        isOpen={isAINotesModalOpen}
        onClose={() => setIsAINotesModalOpen(false)}
        eventTitle={selectedEvent?.title || ''}
        eventDescription={selectedEvent?.extendedProps?.description}
      />
    </motion.div>
  );

  // Renderizado de eventos tipo card + popover animado
  function renderEventContent(eventInfo: any) {
    return (
      <Popover>
        <PopoverTrigger asChild>
          <motion.div
            className={clsx(
              'rounded-xl px-3 py-2 shadow-md font-medium text-sm cursor-pointer flex flex-col gap-1',
              'border border-border',
              'hover:shadow-2xl hover:scale-[1.03] transition-all',
              'bg-white/80 dark:bg-muted/80 backdrop-blur-xl',
              eventInfo.event.backgroundColor ? '' : 'bg-gradient-to-r from-violet-100 to-blue-100 text-primary'
            )}
            style={{ backgroundColor: eventInfo.event.backgroundColor || undefined, color: eventInfo.event.textColor || undefined }}
            whileHover={{ scale: 1.04, boxShadow: '0 4px 24px 0 #6366f155' }}
            whileTap={{ scale: 0.98 }}
          >
            <span className="truncate flex items-center gap-1">
              {eventInfo.event.extendedProps?.location === 'Zoom' && <VideoCamera size={16} className="text-blue-500" />}
              {eventInfo.event.extendedProps?.attendees && <UsersThree size={16} className="text-green-500" />}
              {eventInfo.event.title}
            </span>
            {eventInfo.timeText && <span className="text-xs opacity-70 flex items-center gap-1"><ClockPhosphor size={14} />{eventInfo.timeText}</span>}
          </motion.div>
        </PopoverTrigger>
        <PopoverContent side="top" className="z-50 rounded-2xl bg-white/95 dark:bg-muted/95 px-6 py-4 shadow-2xl border border-border text-sm max-w-xs animate-in fade-in backdrop-blur-xl">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
          >
            <div className="font-bold text-primary mb-1 text-lg flex items-center gap-2">
              <CalendarBlank size={20} className="text-violet-500" />
              {eventInfo.event.title}
            </div>
            {eventInfo.event.extendedProps?.description && (
              <div className="mb-1 text-muted-foreground whitespace-pre-line">{eventInfo.event.extendedProps.description}</div>
            )}
            {eventInfo.event.extendedProps?.location && (
              <div className="flex items-center gap-1 text-xs text-muted-foreground"><MapPin className="h-3 w-3" />{eventInfo.event.extendedProps.location}</div>
            )}
            {eventInfo.event.extendedProps?.zoomLink && (
              <a href={eventInfo.event.extendedProps.zoomLink} target="_blank" rel="noopener noreferrer" className="block mt-2 text-xs text-blue-600 underline">Enlace Zoom</a>
            )}
          </motion.div>
        </PopoverContent>
      </Popover>
    );
  }
}        