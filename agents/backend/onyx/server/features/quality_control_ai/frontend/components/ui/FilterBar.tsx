'use client';

import { memo, type ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { Button } from './Button';
import { X } from 'lucide-react';

interface Filter {
  id: string;
  label: string;
  value: string;
}

interface FilterBarProps {
  filters: Filter[];
  onRemoveFilter: (id: string) => void;
  onClearAll?: () => void;
  className?: string;
  showClearAll?: boolean;
}

const FilterBar = memo(
  ({
    filters,
    onRemoveFilter,
    onClearAll,
    className,
    showClearAll = true,
  }: FilterBarProps): JSX.Element => {
    if (filters.length === 0) {
      return null;
    }

    return (
      <div
        className={cn('flex flex-wrap items-center gap-2 p-3 bg-gray-50 rounded-lg', className)}
        role="group"
        aria-label="Active filters"
      >
        <span className="text-sm text-gray-600 font-medium">Filters:</span>
        {filters.map((filter) => (
          <div
            key={filter.id}
            className="inline-flex items-center gap-1 px-3 py-1 bg-white border border-gray-300 rounded-full text-sm"
          >
            <span className="text-gray-700">{filter.label}:</span>
            <span className="font-medium text-gray-900">{filter.value}</span>
            <button
              type="button"
              onClick={() => onRemoveFilter(filter.id)}
              className="ml-1 text-gray-400 hover:text-gray-600 transition-colors"
              aria-label={`Remove filter ${filter.label}`}
            >
              <X className="w-3 h-3" aria-hidden="true" />
            </button>
          </div>
        ))}
        {showClearAll && onClearAll && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onClearAll}
            className="ml-auto text-sm"
            aria-label="Clear all filters"
          >
            Clear all
          </Button>
        )}
      </div>
    );
  }
);

FilterBar.displayName = 'FilterBar';

export default FilterBar;

