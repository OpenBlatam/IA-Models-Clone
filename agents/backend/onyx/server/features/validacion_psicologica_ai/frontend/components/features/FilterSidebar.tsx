/**
 * Filter sidebar component
 */

'use client';

import React, { useState } from 'react';
import { Drawer, AdvancedFilters, QuickFilters } from '@/components/ui';
import type { FilterState } from './AdvancedFilters';
import type { ValidationStatus } from '@/lib/types';
import { Filter } from 'lucide-react';

export interface FilterSidebarProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  selectedStatuses: ValidationStatus[];
  onStatusToggle: (status: ValidationStatus) => void;
  onClearAll: () => void;
}

export const FilterSidebar: React.FC<FilterSidebarProps> = ({
  filters,
  onFiltersChange,
  selectedStatuses,
  onStatusToggle,
  onClearAll,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setIsOpen(true);
          }
        }}
        className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-accent transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
        aria-label="Abrir panel de filtros"
        tabIndex={0}
      >
        <Filter className="h-4 w-4" aria-hidden="true" />
        <span>Filtros</span>
      </button>

      <Drawer
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title="Filtros"
        position="right"
        size="md"
      >
        <div className="space-y-6 p-4">
          <QuickFilters
            selectedStatuses={selectedStatuses}
            onStatusToggle={onStatusToggle}
            onClear={onClearAll}
          />
          <AdvancedFilters filters={filters} onFiltersChange={onFiltersChange} />
        </div>
      </Drawer>
    </>
  );
};



