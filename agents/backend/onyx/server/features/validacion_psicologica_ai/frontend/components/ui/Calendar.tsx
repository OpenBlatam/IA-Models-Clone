/**
 * Calendar component
 */

'use client';

import React, { useState } from 'react';
import { cn } from '@/lib/utils/cn';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from './Button';
import { format, startOfMonth, endOfMonth, eachDayOfInterval, isSameMonth, isSameDay, addMonths, subMonths } from 'date-fns';

export interface CalendarProps {
  value?: Date;
  onChange?: (date: Date) => void;
  minDate?: Date;
  maxDate?: Date;
  className?: string;
  disabledDates?: Date[];
}

export const Calendar: React.FC<CalendarProps> = ({
  value,
  onChange,
  minDate,
  maxDate,
  className,
  disabledDates = [],
}) => {
  const [currentMonth, setCurrentMonth] = useState(value || new Date());

  const monthStart = startOfMonth(currentMonth);
  const monthEnd = endOfMonth(currentMonth);
  const daysInMonth = eachDayOfInterval({ start: monthStart, end: monthEnd });

  const firstDayOfWeek = monthStart.getDay();
  const daysBeforeMonth = Array.from({ length: firstDayOfWeek }, (_, i) => {
    const date = new Date(monthStart);
    date.setDate(date.getDate() - firstDayOfWeek + i);
    return date;
  });

  const handleDateClick = (date: Date) => {
    if (onChange) {
      onChange(date);
    }
  };

  const isDateDisabled = (date: Date): boolean => {
    if (minDate && date < minDate) {
      return true;
    }
    if (maxDate && date > maxDate) {
      return true;
    }
    return disabledDates.some((disabledDate) => isSameDay(disabledDate, date));
  };

  const handlePreviousMonth = () => {
    setCurrentMonth(subMonths(currentMonth, 1));
  };

  const handleNextMonth = () => {
    setCurrentMonth(addMonths(currentMonth, 1));
  };

  const weekDays = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'];

  return (
    <div className={cn('w-full', className)}>
      <div className="flex items-center justify-between mb-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={handlePreviousMonth}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              handlePreviousMonth();
            }
          }}
          aria-label="Mes anterior"
          tabIndex={0}
        >
          <ChevronLeft className="h-4 w-4" aria-hidden="true" />
        </Button>
        <h3 className="text-lg font-semibold">
          {format(currentMonth, 'MMMM yyyy')}
        </h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleNextMonth}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              handleNextMonth();
            }
          }}
          aria-label="Mes siguiente"
          tabIndex={0}
        >
          <ChevronRight className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>

      <div className="grid grid-cols-7 gap-1">
        {weekDays.map((day) => (
          <div
            key={day}
            className="text-center text-sm font-medium text-muted-foreground py-2"
          >
            {day}
          </div>
        ))}

        {daysBeforeMonth.map((date, index) => (
          <div
            key={`before-${index}`}
            className="aspect-square p-2 text-muted-foreground/30"
          />
        ))}

        {daysInMonth.map((date) => {
          const isSelected = value && isSameDay(date, value);
          const isDisabled = isDateDisabled(date);
          const isCurrentMonth = isSameMonth(date, currentMonth);

          return (
            <button
              key={date.toISOString()}
              type="button"
              onClick={() => !isDisabled && handleDateClick(date)}
              onKeyDown={(e) => {
                if (!isDisabled && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault();
                  handleDateClick(date);
                }
              }}
              disabled={isDisabled}
              className={cn(
                'aspect-square p-2 rounded-md text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
                isSelected && 'bg-primary text-primary-foreground',
                !isSelected && !isDisabled && 'hover:bg-accent',
                isDisabled && 'opacity-30 cursor-not-allowed',
                !isCurrentMonth && 'opacity-50'
              )}
              aria-label={format(date, 'd MMMM yyyy')}
              aria-selected={isSelected}
              aria-disabled={isDisabled}
              tabIndex={isDisabled ? -1 : 0}
            >
              {format(date, 'd')}
            </button>
          );
        })}
      </div>
    </div>
  );
};



