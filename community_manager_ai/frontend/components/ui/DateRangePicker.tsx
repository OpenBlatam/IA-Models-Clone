'use client';

import { useState } from 'react';
import { Calendar, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from './Button';
import { cn } from '@/lib/utils';
import * as Popover from '@radix-ui/react-popover';
import { format, startOfWeek, endOfWeek, startOfMonth, endOfMonth, subDays } from 'date-fns';

interface DateRangePickerProps {
  value?: { start: Date; end: Date };
  onChange: (range: { start: Date; end: Date }) => void;
  className?: string;
}

const presets = [
  { label: 'Hoy', getValue: () => ({ start: new Date(), end: new Date() }) },
  { label: 'Ayer', getValue: () => ({ start: subDays(new Date(), 1), end: subDays(new Date(), 1) }) },
  { label: 'Últimos 7 días', getValue: () => ({ start: subDays(new Date(), 7), end: new Date() }) },
  { label: 'Últimos 30 días', getValue: () => ({ start: subDays(new Date(), 30), end: new Date() }) },
  { label: 'Esta semana', getValue: () => ({ start: startOfWeek(new Date()), end: endOfWeek(new Date()) }) },
  { label: 'Este mes', getValue: () => ({ start: startOfMonth(new Date()), end: endOfMonth(new Date()) }) },
];

export const DateRangePicker = ({ value, onChange, className }: DateRangePickerProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [startDate, setStartDate] = useState(value?.start || subDays(new Date(), 7));
  const [endDate, setEndDate] = useState(value?.end || new Date());

  const handlePreset = (preset: typeof presets[0]) => {
    const range = preset.getValue();
    setStartDate(range.start);
    setEndDate(range.end);
    onChange(range);
    setIsOpen(false);
  };

  const handleApply = () => {
    onChange({ start: startDate, end: endDate });
    setIsOpen(false);
  };

  const formatRange = () => {
    if (!value) return 'Seleccionar rango';
    return `${format(value.start, 'dd MMM')} - ${format(value.end, 'dd MMM')}`;
  };

  return (
    <Popover.Root open={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger asChild>
        <Button
          variant="secondary"
          size="sm"
          className={cn('flex items-center gap-2', className)}
          aria-label="Seleccionar rango de fechas"
        >
          <Calendar className="h-4 w-4" />
          {formatRange()}
        </Button>
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content
          className={cn(
            'z-50 w-80 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg p-4',
            'animate-in fade-in-0 zoom-in-95'
          )}
          sideOffset={5}
          align="start"
        >
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Presets rápidos
              </label>
              <div className="grid grid-cols-2 gap-2">
                {presets.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    onClick={() => handlePreset(preset)}
                    className="rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-left"
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between border-t border-gray-200 dark:border-gray-700 pt-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsOpen(false)}
              >
                Cancelar
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleApply}
              >
                Aplicar
              </Button>
            </div>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
};

