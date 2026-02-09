/**
 * Filter bar component for validations
 */

'use client';

import React, { useState } from 'react';
import { Button, Select, Badge } from '@/components/ui';
import type { ValidationStatus } from '@/lib/types';
import { X, Filter } from 'lucide-react';

export interface FilterBarProps {
  onFilterChange: (filters: FilterState) => void;
}

export interface FilterState {
  status?: ValidationStatus;
  hasProfile?: boolean;
  hasReport?: boolean;
}

const STATUS_OPTIONS = [
  { value: '', label: 'Todos los estados' },
  { value: 'pending', label: 'Pendiente' },
  { value: 'running', label: 'En Proceso' },
  { value: 'completed', label: 'Completada' },
  { value: 'failed', label: 'Fallida' },
  { value: 'cancelled', label: 'Cancelada' },
];

export const FilterBar: React.FC<FilterBarProps> = ({ onFilterChange }) => {
  const [filters, setFilters] = useState<FilterState>({});

  const handleStatusChange = (value: string) => {
    const newFilters = {
      ...filters,
      status: value ? (value as ValidationStatus) : undefined,
    };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const handleProfileToggle = () => {
    const newFilters = {
      ...filters,
      hasProfile: filters.hasProfile ? undefined : true,
    };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const handleReportToggle = () => {
    const newFilters = {
      ...filters,
      hasReport: filters.hasReport ? undefined : true,
    };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const handleClearFilters = () => {
    const clearedFilters: FilterState = {};
    setFilters(clearedFilters);
    onFilterChange(clearedFilters);
  };

  const hasActiveFilters = Object.values(filters).some((value) => value !== undefined);

  return (
    <div className="flex flex-wrap items-center gap-4 p-4 bg-card border rounded-lg">
      <div className="flex items-center gap-2">
        <Filter className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        <span className="text-sm font-medium">Filtros:</span>
      </div>

      <Select
        id="status-filter"
        label=""
        options={STATUS_OPTIONS}
        value={filters.status || ''}
        onChange={(e) => handleStatusChange(e.target.value)}
        className="w-48"
      />

      <Button
        variant={filters.hasProfile ? 'primary' : 'outline'}
        size="sm"
        onClick={handleProfileToggle}
        aria-pressed={!!filters.hasProfile}
      >
        Con Perfil
      </Button>

      <Button
        variant={filters.hasReport ? 'primary' : 'outline'}
        size="sm"
        onClick={handleReportToggle}
        aria-pressed={!!filters.hasReport}
      >
        Con Reporte
      </Button>

      {hasActiveFilters && (
        <Button
          variant="ghost"
          size="sm"
          onClick={handleClearFilters}
          aria-label="Limpiar filtros"
        >
          <X className="h-4 w-4 mr-2" aria-hidden="true" />
          Limpiar
        </Button>
      )}

      {hasActiveFilters && (
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-xs text-muted-foreground">Filtros activos:</span>
          {filters.status && (
            <Badge variant="secondary" className="text-xs">
              {STATUS_OPTIONS.find((opt) => opt.value === filters.status)?.label}
            </Badge>
          )}
          {filters.hasProfile && (
            <Badge variant="secondary" className="text-xs">
              Con Perfil
            </Badge>
          )}
          {filters.hasReport && (
            <Badge variant="secondary" className="text-xs">
              Con Reporte
            </Badge>
          )}
        </div>
      )}
    </div>
  );
};




