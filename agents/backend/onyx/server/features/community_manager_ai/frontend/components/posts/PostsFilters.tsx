/**
 * Posts Filters Component
 * Reusable filter component for posts
 */

'use client';

import { SearchInput } from '@/components/ui/SearchInput';
import { Select } from '@/components/ui/Select';
import { Filter } from 'lucide-react';
import { POST_STATUS } from '@/lib/config/constants';

interface PostsFiltersProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  statusFilter: string;
  onStatusFilterChange: (status: string) => void;
}

/**
 * Posts filters component
 */
export const PostsFilters = ({
  searchQuery,
  onSearchChange,
  statusFilter,
  onStatusFilterChange,
}: PostsFiltersProps) => {
  return (
    <div className="flex flex-col sm:flex-row gap-4">
      <div className="flex-1">
        <SearchInput
          placeholder="Buscar posts..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          onClear={() => onSearchChange('')}
        />
      </div>
      <div className="flex items-center gap-2">
        <Filter className="h-4 w-4 text-gray-500 dark:text-gray-400" aria-hidden="true" />
        <Select
          value={statusFilter}
          onChange={(e) => onStatusFilterChange(e.target.value)}
          options={[
            { value: 'all', label: 'Todos los estados' },
            { value: POST_STATUS.SCHEDULED, label: 'Programados' },
            { value: POST_STATUS.PUBLISHED, label: 'Publicados' },
            { value: POST_STATUS.CANCELLED, label: 'Cancelados' },
          ]}
          className="min-w-[180px]"
          aria-label="Filtrar por estado"
        />
      </div>
    </div>
  );
};


