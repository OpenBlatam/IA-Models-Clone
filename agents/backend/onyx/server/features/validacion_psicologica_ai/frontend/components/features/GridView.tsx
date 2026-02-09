/**
 * Grid view component for validations
 */

'use client';

import React from 'react';
import { ValidationCard } from './ValidationCard';
import { useValidations } from '@/hooks/useValidations';
import { LoadingSpinner, EmptyState, ErrorMessage } from '@/components/ui';
import { FileQuestion } from 'lucide-react';
import { useMemo } from 'react';
import type { FilterState } from './AdvancedFilters';

export interface GridViewProps {
  filters?: FilterState;
  searchTerm?: string;
}

export const GridView: React.FC<GridViewProps> = ({ filters, searchTerm }) => {
  const { data: validations, isLoading, error } = useValidations();

  const filteredValidations = useMemo(() => {
    if (!validations) {
      return [];
    }

    let filtered = validations;

    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      filtered = filtered.filter(
        (validation) =>
          validation.id.toLowerCase().includes(searchLower) ||
          validation.connected_platforms.some((p) => p.toLowerCase().includes(searchLower))
      );
    }

    if (filters) {
      if (filters.status) {
        filtered = filtered.filter((v) => v.status === filters.status);
      }
      if (filters.statuses && filters.statuses.length > 0) {
        filtered = filtered.filter((v) => filters.statuses!.includes(v.status));
      }
      if (filters.hasProfile !== undefined) {
        filtered = filtered.filter((v) => v.has_profile === filters.hasProfile);
      }
      if (filters.hasReport !== undefined) {
        filtered = filtered.filter((v) => v.has_report === filters.hasReport);
      }
      if (filters.dateRange) {
        filtered = filtered.filter((v) => {
          const createdDate = new Date(v.created_at);
          if (filters.dateRange!.start && createdDate < new Date(filters.dateRange!.start)) {
            return false;
          }
          if (filters.dateRange!.end && createdDate > new Date(filters.dateRange!.end)) {
            return false;
          }
          return true;
        });
      }
    }

    return filtered;
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
      <div>
        <ErrorMessage
          message="Error al cargar validaciones. Por favor, intenta de nuevo."
          title="Error"
        />
      </div>
    );
  }

  if (!filteredValidations || filteredValidations.length === 0) {
    return (
      <div>
        <EmptyState
          title="No hay validaciones"
          description="Crea una nueva validación para comenzar."
          icon={<FileQuestion className="h-12 w-12 text-muted-foreground" />}
        />
      </div>
    );
  }

  return (
    <div
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
      role="list"
      aria-label="Lista de validaciones en vista de grid"
    >
      {filteredValidations.map((validation) => (
        <ValidationCard key={validation.id} validation={validation} />
      ))}
    </div>
  );
};
