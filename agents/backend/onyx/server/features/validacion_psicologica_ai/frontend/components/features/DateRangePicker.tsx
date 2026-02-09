/**
 * Date range picker component
 */

'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, Calendar, Button } from '@/components/ui';
import { Popover } from '@/components/ui';
import { Calendar as CalendarIcon, X } from 'lucide-react';
import { format } from 'date-fns';

export interface DateRangePickerProps {
  startDate?: Date;
  endDate?: Date;
  onRangeChange: (start?: Date, end?: Date) => void;
  minDate?: Date;
  maxDate?: Date;
  className?: string;
}

export const DateRangePicker: React.FC<DateRangePickerProps> = ({
  startDate,
  endDate,
  onRangeChange,
  minDate,
  maxDate,
  className,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [tempStartDate, setTempStartDate] = useState<Date | undefined>(startDate);
  const [tempEndDate, setTempEndDate] = useState<Date | undefined>(endDate);

  const handleStartDateSelect = (date: Date) => {
    setTempStartDate(date);
    if (tempEndDate && date > tempEndDate) {
      setTempEndDate(undefined);
    }
  };

  const handleEndDateSelect = (date: Date) => {
    if (tempStartDate && date < tempStartDate) {
      return;
    }
    setTempEndDate(date);
  };

  const handleApply = () => {
    onRangeChange(tempStartDate, tempEndDate);
    setIsOpen(false);
  };

  const handleClear = () => {
    setTempStartDate(undefined);
    setTempEndDate(undefined);
    onRangeChange(undefined, undefined);
    setIsOpen(false);
  };

  const displayText = startDate && endDate
    ? `${format(startDate, 'dd/MM/yyyy')} - ${format(endDate, 'dd/MM/yyyy')}`
    : startDate
    ? `Desde ${format(startDate, 'dd/MM/yyyy')}`
    : endDate
    ? `Hasta ${format(endDate, 'dd/MM/yyyy')}`
    : 'Seleccionar rango de fechas';

  return (
    <Popover
      trigger={
        <Button variant="outline" className={className}>
          <CalendarIcon className="h-4 w-4 mr-2" aria-hidden="true" />
          {displayText}
        </Button>
      }
      content={
        <Card className="w-auto">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Seleccionar Rango</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleClear}
                aria-label="Limpiar fechas"
                tabIndex={0}
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-medium mb-2">Fecha Inicio</h4>
                <Calendar
                  value={tempStartDate}
                  onChange={handleStartDateSelect}
                  minDate={minDate}
                  maxDate={tempEndDate || maxDate}
                />
              </div>
              <div>
                <h4 className="text-sm font-medium mb-2">Fecha Fin</h4>
                <Calendar
                  value={tempEndDate}
                  onChange={handleEndDateSelect}
                  minDate={tempStartDate || minDate}
                  maxDate={maxDate}
                />
              </div>
            </div>
            <div className="flex items-center justify-end gap-2 pt-4 border-t">
              <Button variant="outline" onClick={() => setIsOpen(false)}>
                Cancelar
              </Button>
              <Button onClick={handleApply}>
                Aplicar
              </Button>
            </div>
          </CardContent>
        </Card>
      }
      isOpen={isOpen}
      onOpenChange={setIsOpen}
      position="bottom"
      align="start"
    />
  );
};



