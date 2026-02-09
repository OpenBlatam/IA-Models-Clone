/**
 * Component to display list of validations
 */

'use client';

import React from 'react';
import { useValidations } from '@/hooks/useValidations';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  LoadingSpinner,
  Badge,
  EmptyState,
  ErrorMessage,
  Skeleton,
} from '@/components/ui';
import { ValidationCard } from './ValidationCard';
import { ValidationTable } from './ValidationTable';
import { GridView } from './GridView';
import { TimelineView } from './TimelineView';
import { AdvancedFilters } from './AdvancedFilters';
import { FilterChips } from './FilterChips';
import { ViewSwitcher } from './ViewSwitcher';
import { SearchInput } from '@/components/ui';
import { useViewType } from '@/hooks/useViewType';
import type { ValidationStatus, ValidationRead } from '@/lib/types';
import type { FilterState } from './AdvancedFilters';
import { FileQuestion, CheckCircle2, XCircle, Play, Clock } from 'lucide-react';
import { useState, useMemo } from 'react';

const getStatusIcon = (status: ValidationStatus): React.ReactNode => {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="h-5 w-5 text-green-500" aria-hidden="true" />;
    case 'failed':
      return <XCircle className="h-5 w-5 text-red-500" aria-hidden="true" />;
    case 'running':
      return <Play className="h-5 w-5 text-blue-500 animate-pulse" aria-hidden="true" />;
    default:
      return <Clock className="h-5 w-5 text-gray-500" aria-hidden="true" />;
  }
};

const getStatusLabel = (status: ValidationStatus): string => {
  const labels: Record<ValidationStatus, string> = {
    pending: 'Pendiente',
    running: 'En Proceso',
    completed: 'Completada',
    failed: 'Fallida',
    cancelled: 'Cancelada',
  };
  return labels[status];
};

const getStatusVariant = (status: ValidationStatus): 'success' | 'destructive' | 'warning' | 'info' | 'default' => {
  switch (status) {
    case 'completed':
      return 'success';
    case 'failed':
      return 'destructive';
    case 'running':
      return 'info';
    case 'cancelled':
      return 'warning';
    default:
      return 'default';
  }
};

export const ValidationList: React.FC = () => {
  const { data: validations, isLoading, error } = useValidations();
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState<FilterState>({});
  const [viewType, setViewType] = useViewType('list');

  const filteredValidations = useMemo(() => {
    if (!validations) {
      return [];
    }

    return validations.filter((validation) => {
      if (searchTerm) {
        const searchLower = searchTerm.toLowerCase();
        const matchesSearch =
          validation.id.toLowerCase().includes(searchLower) ||
          validation.connected_platforms.some((p) =>
            p.toLowerCase().includes(searchLower)
          );
        if (!matchesSearch) {
          return false;
        }
      }

      if (filters.status && validation.status !== filters.status) {
        return false;
      }

      if (filters.statuses && filters.statuses.length > 0) {
        if (!filters.statuses.includes(validation.status)) {
          return false;
        }
      }

      if (filters.hasProfile !== undefined && validation.has_profile !== filters.hasProfile) {
        return false;
      }

      if (filters.hasReport !== undefined && validation.has_report !== filters.hasReport) {
        return false;
      }

      if (filters.dateRange) {
        const createdDate = new Date(validation.created_at);
        if (filters.dateRange.start && createdDate < new Date(filters.dateRange.start)) {
          return false;
        }
        if (filters.dateRange.end && createdDate > new Date(filters.dateRange.end)) {
          return false;
        }
      }

      return true;
    });
  }, [validations, searchTerm, filters]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-12" role="status" aria-live="polite">
        <LoadingSpinner size="lg" />
        <span className="sr-only">Cargando validaciones...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <ErrorMessage
            message="Error al cargar validaciones. Por favor, intenta de nuevo."
            title="Error"
          />
        </CardContent>
      </Card>
    );
  }

  if (!validations || validations.length === 0) {
    return (
      <Card>
        <CardContent>
          <EmptyState
            title="No hay validaciones"
            description="Crea una nueva validación para comenzar el análisis psicológico."
            icon={<FileQuestion className="h-12 w-12 text-muted-foreground" />}
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="space-y-4">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex-1 max-w-md">
            <SearchInput
              value={searchTerm}
              onChange={setSearchTerm}
              placeholder="Buscar validaciones..."
            />
          </div>
          <ViewSwitcher currentView={viewType} onViewChange={setViewType} />
        </div>

        <AdvancedFilters filters={filters} onFiltersChange={setFilters} />
      </div>

      {Object.keys(filters).length > 0 && (
        <FilterChips
          filters={filters}
          onRemove={(key) => {
            const newFilters = { ...filters };
            if (key === 'statuses') {
              newFilters.statuses = [];
            } else {
              delete newFilters[key];
            }
            setFilters(newFilters);
          }}
          onClearAll={() => setFilters({})}
        />
      )}

      {isLoading ? (
        <div className="space-y-4" role="status" aria-live="polite">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-6 w-48" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-4 w-3/4" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : filteredValidations.length === 0 ? (
        <Card>
          <CardContent>
            <EmptyState
              title={
                searchTerm || Object.keys(filters).length > 0
                  ? 'No se encontraron validaciones'
                  : 'No hay validaciones'
              }
              description={
                searchTerm || Object.keys(filters).length > 0
                  ? 'Intenta ajustar tus filtros o búsqueda'
                  : 'Crea una nueva validación para comenzar'
              }
              icon={<FileQuestion className="h-12 w-12 text-muted-foreground" />}
            />
          </CardContent>
        </Card>
      ) : viewType === 'table' ? (
        <ValidationTable />
      ) : viewType === 'grid' ? (
        <GridView filters={filters} searchTerm={searchTerm} />
      ) : viewType === 'timeline' ? (
        <TimelineView />
      ) : (
        <div className="space-y-4" role="list" aria-label="Lista de validaciones">
          {filteredValidations.map((validation) => (
            <ValidationCard key={validation.id} validation={validation} />
          ))}
        </div>
      )}
    </div>
  );
};

