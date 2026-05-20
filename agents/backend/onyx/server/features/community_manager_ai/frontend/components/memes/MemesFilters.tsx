/**
 * Memes Filters Component
 * Reusable filter component for memes
 */

'use client';

import { SearchInput } from '@/components/ui/SearchInput';
import { Select } from '@/components/ui/Select';
import { Search } from 'lucide-react';

interface MemesFiltersProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  selectedCategory: string;
  onCategoryChange: (category: string) => void;
  categories: string[];
}

/**
 * Memes filters component
 */
export const MemesFilters = ({
  searchQuery,
  onSearchChange,
  selectedCategory,
  onCategoryChange,
  categories,
}: MemesFiltersProps) => {
  return (
    <div className="flex flex-col sm:flex-row gap-4">
      <div className="relative flex-1 max-w-md">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400 dark:text-gray-500" aria-hidden="true" />
        <SearchInput
          placeholder="Buscar memes..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          onClear={() => onSearchChange('')}
          className="pl-10"
        />
      </div>
      <Select
        value={selectedCategory}
        onChange={(e) => onCategoryChange(e.target.value)}
        options={[
          { value: '', label: 'Todas las categorías' },
          ...categories.map((cat) => ({ value: cat, label: cat })),
        ]}
        className="min-w-[200px]"
        aria-label="Filtrar por categoría"
      />
    </div>
  );
};


