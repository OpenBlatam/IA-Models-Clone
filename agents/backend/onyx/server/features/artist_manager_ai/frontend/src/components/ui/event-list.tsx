'use client';

import Link from 'next/link';
import { formatDate, formatTime } from '@/lib/utils';
import { CalendarEvent } from '@/types';
import { cn } from '@/lib/utils';
import { Calendar as CalendarIcon } from 'lucide-react';

interface EventListProps {
  events: CalendarEvent[];
  maxItems?: number;
  showDate?: boolean;
  showTime?: boolean;
  emptyMessage?: string;
  className?: string;
}

const EventList = ({
  events,
  maxItems = 5,
  showDate = true,
  showTime = true,
  emptyMessage = 'No hay eventos',
  className,
}: EventListProps) => {
  if (events.length === 0) {
    return (
      <div className={cn('text-center py-8 text-gray-500', className)}>
        <CalendarIcon className="w-12 h-12 mx-auto mb-2 text-gray-300" />
        <p>{emptyMessage}</p>
      </div>
    );
  }

  const displayEvents = events.slice(0, maxItems);

  return (
    <ul className={cn('space-y-3', className)}>
      {displayEvents.map((event) => (
        <li key={event.id} className="border-b pb-3 last:border-b-0">
          <Link
            href={`/calendar/${event.id}`}
            className="block hover:bg-gray-50 p-2 rounded transition-colors group"
          >
            <p className="font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
              {event.title}
            </p>
            <div className="flex items-center gap-2 mt-1 text-sm text-gray-600">
              {showDate && <span>{formatDate(event.start_time)}</span>}
              {showDate && showTime && <span>•</span>}
              {showTime && <span>{formatTime(event.start_time)}</span>}
            </div>
          </Link>
        </li>
      ))}
    </ul>
  );
};

export { EventList };

