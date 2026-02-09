/**
 * Date range filter component
 */

'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, Button, DatePicker } from '@/components/ui';
import { Calendar, X } from 'lucide-react';
import { format } from 'date-fns';

export interface DateRange {
  start?: string;
  end?: string;
}

export interface DateRangeFilterProps {
  onRangeChange: (range: DateRange) => void;
  defaultRange?: DateRange;
}

export const DateRangeFilter: React.FC<DateRangeFilterProps> = ({
  onRangeChange,
  defaultRange,
}) => {
  const [range, setRange] = useState<DateRange>(defaultRange || {});
  const [isOpen, setIsOpen] = useState(false);

  const handleStartChange = (value: string) => {
    const newRange = { ...range, start: value };
    setRange(newRange);
    onRangeChange(newRange);
  };

  const handleEndChange = (value: string) => {
    const newRange = { ...range, end: value };
    setRange(newRange);
    onRangeChange(newRange);
  };

  const handleClear = () => {
    const clearedRange: DateRange = {};
    setRange(clearedRange);
    onRangeChange(clearedRange);
  };

  const hasActiveFilter = range.start || range.end;

  return (
    <div className="relative">
      <Button
        variant="outline"
        size="sm"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-label="Filtro de rango de fechas"
        tabIndex={0}
      >
        <Calendar className="h-4 w-4 mr-2" aria-hidden="true" />
        Rango de Fechas
        {hasActiveFilter && (
          <span className="ml-2 h-2 w-2 rounded-full bg-primary" aria-hidden="true" />
        )}
      </Button>

      {isOpen && (
        <Card className="absolute right-0 mt-2 w-80 z-50 shadow-lg">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm">Filtro de Fechas</CardTitle>
            <div className="flex items-center gap-2">
              {hasActiveFilter && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClear}
                  aria-label="Limpiar filtro"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsOpen(false)}
                aria-label="Cerrar"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <DatePicker
              label="Fecha Inicio"
              value={range.start}
              onChange={handleStartChange}
              max={range.end}
            />
            <DatePicker
              label="Fecha Fin"
              value={range.end}
              onChange={handleEndChange}
              min={range.start}
            />
            {hasActiveFilter && (
              <div className="pt-2 border-t text-xs text-muted-foreground">
                {range.start && (
                  <p>Desde: {format(new Date(range.start), 'PP')}</p>
                )}
                {range.end && (
                  <p>Hasta: {format(new Date(range.end), 'PP')}</p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};




