/**
 * Validation filters accordion component
 */

'use client';

import React from 'react';
import { Accordion } from '@/components/ui';
import { StatusFilter } from './StatusFilter';
import { DateRangeFilter } from './DateRangeFilter';
import { Filter } from 'lucide-react';
import type { ValidationStatus } from '@/lib/types';
import type { DateRange } from './DateRangeFilter';

export interface ValidationFiltersAccordionProps {
  selectedStatuses: ValidationStatus[];
  onStatusToggle: (status: ValidationStatus) => void;
  onStatusClear: () => void;
  dateRange?: DateRange;
  onDateRangeChange: (range?: DateRange) => void;
}

export const ValidationFiltersAccordion: React.FC<ValidationFiltersAccordionProps> = ({
  selectedStatuses,
  onStatusToggle,
  onStatusClear,
  dateRange,
  onDateRangeChange,
}) => {
  const items = [
    {
      id: 'status',
      title: 'Filtros por Estado',
      icon: <Filter className="h-4 w-4" aria-hidden="true" />,
      content: (
        <StatusFilter
          selectedStatuses={selectedStatuses}
          onStatusToggle={onStatusToggle}
          onClear={onStatusClear}
        />
      ),
    },
    {
      id: 'date',
      title: 'Filtros por Fecha',
      icon: <Filter className="h-4 w-4" aria-hidden="true" />,
      content: (
        <DateRangeFilter
          onRangeChange={onDateRangeChange}
          defaultRange={dateRange}
        />
      ),
    },
  ];

  return <Accordion items={items} allowMultiple defaultOpen={[]} />;
};



