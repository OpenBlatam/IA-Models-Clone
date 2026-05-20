'use client';

import React, { memo, useMemo } from 'react';
import { Filter, X } from 'lucide-react';
import { Button } from './Button';
import { Select } from './Select';
import { clsx } from 'clsx';

export interface FilterOption {
  label: string;
  value: string;
}

interface FilterBarProps {
  filters: {
    label: string;
    options: FilterOption[];
    value: string;
    onChange: (value: string) => void;
  }[];
  onClearAll?: () => void;
  className?: string;
}

export const FilterBar: React.FC<FilterBarProps> = memo(({
  filters,
  onClearAll,
  className,
}) => {
  const hasActiveFilters = useMemo(
    () => filters.some((f) => f.value !== ''),
    [filters]
  );

  return (
    <div
      className={clsx(
        'flex flex-wrap items-center gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700',
        className
      )}
    >
      <div className="flex items-center space-x-2 text-gray-700 dark:text-gray-300">
        <Filter className="h-4 w-4" />
        <span className="text-sm font-medium">Filters:</span>
      </div>

      {filters.map((filter, index) => (
        <div key={index} className="flex items-center space-x-2">
          <label className="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">
            {filter.label}:
          </label>
          <Select
            value={filter.value}
            onChange={(e) => filter.onChange(e.target.value)}
            options={[
              { value: '', label: 'All' },
              ...filter.options,
            ]}
            className="min-w-[120px]"
          />
        </div>
      ))}

      {hasActiveFilters && onClearAll && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onClearAll}
          className="ml-auto"
        >
          <X className="h-4 w-4 mr-1" />
          Clear Filters
        </Button>
      )}
    </div>
  );
});

FilterBar.displayName = 'FilterBar';

