/**
 * Advanced validation filters component
 */

'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, FormField, Select, Toggle, Button } from '@/components/ui';
import { Filter, X } from 'lucide-react';
import type { ValidationStatus } from '@/lib/types';

export interface ValidationFiltersAdvancedProps {
  onFiltersChange: (filters: FilterState) => void;
  className?: string;
}

export interface FilterState {
  status?: ValidationStatus;
  hasProfile?: boolean;
  hasReport?: boolean;
  platform?: string;
  dateRange?: {
    start?: string;
    end?: string;
  };
}

const STATUS_OPTIONS = [
  { value: 'pending', label: 'Pendiente' },
  { value: 'running', label: 'En Proceso' },
  { value: 'completed', label: 'Completada' },
  { value: 'failed', label: 'Fallida' },
  { value: 'cancelled', label: 'Cancelada' },
];

const PLATFORM_OPTIONS = [
  { value: 'facebook', label: 'Facebook' },
  { value: 'twitter', label: 'Twitter/X' },
  { value: 'instagram', label: 'Instagram' },
  { value: 'linkedin', label: 'LinkedIn' },
  { value: 'tiktok', label: 'TikTok' },
  { value: 'youtube', label: 'YouTube' },
  { value: 'reddit', label: 'Reddit' },
  { value: 'discord', label: 'Discord' },
  { value: 'telegram', label: 'Telegram' },
];

export const ValidationFiltersAdvanced: React.FC<ValidationFiltersAdvancedProps> = ({
  onFiltersChange,
  className,
}) => {
  const [filters, setFilters] = useState<FilterState>({});

  const handleFilterChange = (key: keyof FilterState, value: any) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    onFiltersChange(newFilters);
  };

  const handleClear = () => {
    const clearedFilters: FilterState = {};
    setFilters(clearedFilters);
    onFiltersChange(clearedFilters);
  };

  const activeFiltersCount = Object.values(filters).filter(
    (v) => v !== undefined && (typeof v === 'object' ? Object.keys(v).length > 0 : true)
  ).length;

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" aria-hidden="true" />
            Filtros Avanzados
          </CardTitle>
          {activeFiltersCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleClear();
                }
              }}
              aria-label="Limpiar filtros"
              tabIndex={0}
            >
              <X className="h-4 w-4 mr-2" aria-hidden="true" />
              Limpiar ({activeFiltersCount})
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <FormField label="Estado">
          <Select
            options={STATUS_OPTIONS}
            value={filters.status}
            onChange={(value) => handleFilterChange('status', value)}
            placeholder="Todos los estados"
          />
        </FormField>

        <FormField label="Plataforma">
          <Select
            options={PLATFORM_OPTIONS}
            value={filters.platform}
            onChange={(value) => handleFilterChange('platform', value)}
            placeholder="Todas las plataformas"
          />
        </FormField>

        <div className="space-y-3">
          <FormField label="Opciones">
            <div className="space-y-2">
              <Toggle
                checked={filters.hasProfile === true}
                onChange={(checked) =>
                  handleFilterChange('hasProfile', checked ? true : undefined)
                }
                label="Solo con perfil"
                id="has-profile"
              />
              <Toggle
                checked={filters.hasReport === true}
                onChange={(checked) =>
                  handleFilterChange('hasReport', checked ? true : undefined)
                }
                label="Solo con reporte"
                id="has-report"
              />
            </div>
          </FormField>
        </div>
      </CardContent>
    </Card>
  );
};



