/**
 * Quick filters component
 */

'use client';

import React from 'react';
import { Button, Badge } from '@/components/ui';
import { Filter, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import type { ValidationStatus } from '@/lib/types';

export interface QuickFiltersProps {
  selectedStatuses: ValidationStatus[];
  onStatusToggle: (status: ValidationStatus) => void;
  onClear: () => void;
  className?: string;
}

const STATUS_OPTIONS: Array<{ value: ValidationStatus; label: string; variant: 'success' | 'destructive' | 'warning' | 'info' | 'default' }> = [
  { value: 'completed', label: 'Completadas', variant: 'success' },
  { value: 'running', label: 'En Proceso', variant: 'info' },
  { value: 'pending', label: 'Pendientes', variant: 'default' },
  { value: 'failed', label: 'Fallidas', variant: 'destructive' },
  { value: 'cancelled', label: 'Canceladas', variant: 'warning' },
];

export const QuickFilters: React.FC<QuickFiltersProps> = ({
  selectedStatuses,
  onStatusToggle,
  onClear,
  className,
}) => {
  const handleToggle = (status: ValidationStatus) => {
    onStatusToggle(status);
  };

  const handleKeyDown = (event: React.KeyboardEvent, status: ValidationStatus) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleToggle(status);
    }
  };

  return (
    <div className={cn('flex items-center gap-2 flex-wrap', className)}>
      <div className="flex items-center gap-2">
        <Filter className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        <span className="text-sm font-medium">Filtros rápidos:</span>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        {STATUS_OPTIONS.map((option) => {
          const isSelected = selectedStatuses.includes(option.value);

          return (
            <Button
              key={option.value}
              variant={isSelected ? 'primary' : 'outline'}
              size="sm"
              onClick={() => handleToggle(option.value)}
              onKeyDown={(e) => handleKeyDown(e, option.value)}
              aria-pressed={isSelected}
              aria-label={`Filtrar por ${option.label}`}
              tabIndex={0}
            >
              {option.label}
              {isSelected && (
                <Badge variant={option.variant} className="ml-2">
                  {selectedStatuses.filter((s) => s === option.value).length}
                </Badge>
              )}
            </Button>
          );
        })}
      </div>

      {selectedStatuses.length > 0 && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onClear}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onClear();
            }
          }}
          aria-label="Limpiar filtros"
          tabIndex={0}
        >
          <X className="h-4 w-4 mr-2" aria-hidden="true" />
          Limpiar
        </Button>
      )}
    </div>
  );
};



