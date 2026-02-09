/**
 * Filter chips component for active filters
 */

'use client';

import React from 'react';
import { Chip } from '@/components/ui';
import { X } from 'lucide-react';
import type { FilterState } from './AdvancedFilters';
import type { ValidationStatus } from '@/lib/types';

export interface FilterChipsProps {
  filters: FilterState;
  onRemove: (key: keyof FilterState) => void;
  onClearAll: () => void;
  className?: string;
}

const STATUS_LABELS: Record<ValidationStatus, string> = {
  pending: 'Pendiente',
  running: 'En Proceso',
  completed: 'Completada',
  failed: 'Fallida',
  cancelled: 'Cancelada',
};

export const FilterChips: React.FC<FilterChipsProps> = ({
  filters,
  onRemove,
  onClearAll,
  className,
}) => {
  const activeFilters = React.useMemo(() => {
    const chips: Array<{ key: keyof FilterState; label: string; value: string }> = [];

    if (filters.status) {
      chips.push({
        key: 'status',
        label: 'Estado',
        value: STATUS_LABELS[filters.status] || filters.status,
      });
    }

    if (filters.statuses && filters.statuses.length > 0) {
      filters.statuses.forEach((status) => {
        chips.push({
          key: 'statuses',
          label: 'Estado',
          value: STATUS_LABELS[status] || status,
        });
      });
    }

    if (filters.hasProfile !== undefined) {
      chips.push({
        key: 'hasProfile',
        label: 'Perfil',
        value: filters.hasProfile ? 'Con perfil' : 'Sin perfil',
      });
    }

    if (filters.hasReport !== undefined) {
      chips.push({
        key: 'hasReport',
        label: 'Reporte',
        value: filters.hasReport ? 'Con reporte' : 'Sin reporte',
      });
    }

    if (filters.dateRange) {
      if (filters.dateRange.start && filters.dateRange.end) {
        chips.push({
          key: 'dateRange',
          label: 'Fecha',
          value: `${filters.dateRange.start} - ${filters.dateRange.end}`,
        });
      } else if (filters.dateRange.start) {
        chips.push({
          key: 'dateRange',
          label: 'Desde',
          value: filters.dateRange.start,
        });
      } else if (filters.dateRange.end) {
        chips.push({
          key: 'dateRange',
          label: 'Hasta',
          value: filters.dateRange.end,
        });
      }
    }

    return chips;
  }, [filters]);

  if (activeFilters.length === 0) {
    return null;
  }

  return (
    <div className={`flex items-center gap-2 flex-wrap ${className || ''}`}>
      <span className="text-sm text-muted-foreground">Filtros activos:</span>
      {activeFilters.map((filter, index) => (
        <Chip
          key={`${filter.key}-${index}`}
          label={`${filter.label}: ${filter.value}`}
          onRemove={() => onRemove(filter.key)}
          variant="outline"
          size="sm"
        />
      ))}
      {activeFilters.length > 1 && (
        <button
          type="button"
          onClick={onClearAll}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onClearAll();
            }
          }}
          className="text-xs text-muted-foreground hover:text-foreground underline focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded"
          aria-label="Limpiar todos los filtros"
          tabIndex={0}
        >
          Limpiar todo
        </button>
      )}
    </div>
  );
};



