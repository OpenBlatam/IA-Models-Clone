/**
 * Advanced filters component
 */

'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, Button, Accordion } from '@/components/ui';
import { FilterBar } from './FilterBar';
import { DateRangeFilter } from './DateRangeFilter';
import { StatusFilter } from './StatusFilter';
import { Filter, X } from 'lucide-react';
import type { ValidationStatus } from '@/lib/types';
import type { DateRange } from './DateRangeFilter';

export interface AdvancedFiltersProps {
  filters?: FilterState;
  onFiltersChange: (filters: FilterState) => void;
}

export interface FilterState {
  status?: ValidationStatus;
  hasProfile?: boolean;
  hasReport?: boolean;
  dateRange?: DateRange;
  statuses?: ValidationStatus[];
}

export const AdvancedFilters: React.FC<AdvancedFiltersProps> = ({
  filters: externalFilters,
  onFiltersChange,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [filters, setFilters] = useState<FilterState>(externalFilters || {});
  const [selectedStatuses, setSelectedStatuses] = useState<ValidationStatus[]>(
    externalFilters?.statuses || []
  );

  React.useEffect(() => {
    if (externalFilters) {
      setFilters(externalFilters);
      setSelectedStatuses(externalFilters.statuses || []);
    }
  }, [externalFilters]);

  const handleFilterChange = (newFilters: Partial<FilterState>) => {
    const updatedFilters = { ...filters, ...newFilters };
    setFilters(updatedFilters);
    onFiltersChange(updatedFilters);
  };

  const handleStatusToggle = (status: ValidationStatus) => {
    const newStatuses = selectedStatuses.includes(status)
      ? selectedStatuses.filter((s) => s !== status)
      : [...selectedStatuses, status];
    setSelectedStatuses(newStatuses);
    handleFilterChange({ statuses: newStatuses });
  };

  const handleClearAll = () => {
    const clearedFilters: FilterState = {};
    setFilters(clearedFilters);
    setSelectedStatuses([]);
    onFiltersChange(clearedFilters);
  };

  const activeFiltersCount = Object.values(filters).filter((v) => v !== undefined && (Array.isArray(v) ? v.length > 0 : true)).length;

  const accordionItems = [
    {
      id: 'status',
      title: 'Filtros por Estado',
      content: (
        <StatusFilter
          selectedStatuses={selectedStatuses}
          onStatusToggle={handleStatusToggle}
          onClear={() => {
            setSelectedStatuses([]);
            handleFilterChange({ statuses: [] });
          }}
        />
      ),
    },
    {
      id: 'date',
      title: 'Filtros por Fecha',
      content: (
        <DateRangeFilter
          onRangeChange={(range) => handleFilterChange({ dateRange: range })}
          defaultRange={filters.dateRange}
        />
      ),
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          aria-label="Filtros avanzados"
          tabIndex={0}
        >
          <Filter className="h-4 w-4 mr-2" aria-hidden="true" />
          Filtros Avanzados
          {activeFiltersCount > 0 && (
            <span className="ml-2 h-5 w-5 rounded-full bg-primary text-primary-foreground text-xs flex items-center justify-center">
              {activeFiltersCount}
            </span>
          )}
        </Button>
        {activeFiltersCount > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClearAll}
            aria-label="Limpiar todos los filtros"
            tabIndex={0}
          >
            <X className="h-4 w-4 mr-2" aria-hidden="true" />
            Limpiar Todo
          </Button>
        )}
      </div>

      {isOpen && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Filtros Avanzados</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <FilterBar onFilterChange={(f) => handleFilterChange(f)} />
            <Accordion items={accordionItems} allowMultiple defaultOpen={[]} />
          </CardContent>
        </Card>
      )}
    </div>
  );
};


