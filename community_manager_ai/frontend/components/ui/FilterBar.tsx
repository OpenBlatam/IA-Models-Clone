'use client';

import { useState } from 'react';
import { X, Filter } from 'lucide-react';
import { Button } from './Button';
import { Badge } from './Badge';
import { cn } from '@/lib/utils';

interface Filter {
  id: string;
  label: string;
  value: string;
}

interface FilterBarProps {
  filters: Filter[];
  onRemoveFilter: (id: string) => void;
  onClearAll: () => void;
  className?: string;
}

export const FilterBar = ({ filters, onRemoveFilter, onClearAll, className }: FilterBarProps) => {
  if (filters.length === 0) return null;

  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-3',
        className
      )}
    >
      <div className="flex items-center gap-2">
        <Filter className="h-4 w-4 text-gray-500 dark:text-gray-400" />
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Filtros:</span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {filters.map((filter) => (
          <Badge
            key={filter.id}
            variant="default"
            className="flex items-center gap-1 pr-1"
          >
            <span>{filter.label}: {filter.value}</span>
            <button
              type="button"
              onClick={() => onRemoveFilter(filter.id)}
              className="ml-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 p-0.5 transition-colors"
              aria-label={`Eliminar filtro ${filter.label}`}
            >
              <X className="h-3 w-3" />
            </button>
          </Badge>
        ))}
      </div>
      <Button
        variant="ghost"
        size="sm"
        onClick={onClearAll}
        className="ml-auto"
        aria-label="Limpiar todos los filtros"
      >
        Limpiar todo
      </Button>
    </div>
  );
};



