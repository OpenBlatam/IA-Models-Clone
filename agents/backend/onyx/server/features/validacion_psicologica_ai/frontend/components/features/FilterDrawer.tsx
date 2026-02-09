/**
 * Filter drawer component
 */

'use client';

import React from 'react';
import { Drawer } from '@/components/ui';
import { AdvancedFilters } from './AdvancedFilters';
import type { FilterState } from './AdvancedFilters';

export interface FilterDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
}

export const FilterDrawer: React.FC<FilterDrawerProps> = ({
  isOpen,
  onClose,
  filters,
  onFiltersChange,
}) => {
  return (
    <Drawer
      isOpen={isOpen}
      onClose={onClose}
      title="Filtros Avanzados"
      position="right"
      size="md"
    >
      <div className="p-4">
        <AdvancedFilters filters={filters} onFiltersChange={onFiltersChange} />
      </div>
    </Drawer>
  );
};

