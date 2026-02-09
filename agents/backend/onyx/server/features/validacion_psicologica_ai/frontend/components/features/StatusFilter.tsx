/**
 * Status filter component
 */

'use client';

import React from 'react';
import { Button, Badge } from '@/components/ui';
import type { ValidationStatus } from '@/lib/types';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

export interface StatusFilterProps {
  selectedStatuses: ValidationStatus[];
  onStatusToggle: (status: ValidationStatus) => void;
  onClear: () => void;
}

const STATUS_OPTIONS: { value: ValidationStatus; label: string; color: string }[] = [
  { value: 'pending', label: 'Pendiente', color: 'bg-gray-500' },
  { value: 'running', label: 'En Proceso', color: 'bg-blue-500' },
  { value: 'completed', label: 'Completada', color: 'bg-green-500' },
  { value: 'failed', label: 'Fallida', color: 'bg-red-500' },
  { value: 'cancelled', label: 'Cancelada', color: 'bg-yellow-500' },
];

export const StatusFilter: React.FC<StatusFilterProps> = ({
  selectedStatuses,
  onStatusToggle,
  onClear,
}) => {
  const handleKeyDown = (event: React.KeyboardEvent, status: ValidationStatus) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onStatusToggle(status);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-sm font-medium text-muted-foreground">Estado:</span>
      {STATUS_OPTIONS.map((option) => {
        const isSelected = selectedStatuses.includes(option.value);
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onStatusToggle(option.value)}
            onKeyDown={(e) => handleKeyDown(e, option.value)}
            className={cn(
              'px-3 py-1.5 text-sm rounded-md border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              isSelected
                ? 'bg-primary text-primary-foreground border-primary'
                : 'bg-background border-input hover:bg-accent'
            )}
            aria-pressed={isSelected}
            aria-label={`${isSelected ? 'Deseleccionar' : 'Seleccionar'} ${option.label}`}
            tabIndex={0}
          >
            <div className="flex items-center gap-2">
              <span
                className={cn('h-2 w-2 rounded-full', option.color)}
                aria-hidden="true"
              />
              {option.label}
            </div>
          </button>
        );
      })}
      {selectedStatuses.length > 0 && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onClear}
          aria-label="Limpiar filtros de estado"
          tabIndex={0}
        >
          <X className="h-4 w-4 mr-1" aria-hidden="true" />
          Limpiar
        </Button>
      )}
    </div>
  );
};




