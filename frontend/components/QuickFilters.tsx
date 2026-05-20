'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FiFilter, FiX } from 'react-icons/fi';

interface QuickFilter {
  id: string;
  label: string;
  value: string;
  count?: number;
}

interface QuickFiltersProps {
  filters: QuickFilter[];
  selected: string[];
  onChange: (selected: string[]) => void;
  maxVisible?: number;
}

export default function QuickFilters({
  filters,
  selected,
  onChange,
  maxVisible = 5,
}: QuickFiltersProps) {
  const [showAll, setShowAll] = useState(false);
  const visibleFilters = showAll ? filters : filters.slice(0, maxVisible);

  const toggleFilter = (filterId: string) => {
    if (selected.includes(filterId)) {
      onChange(selected.filter((id) => id !== filterId));
    } else {
      onChange([...selected, filterId]);
    }
  };

  const clearAll = () => {
    onChange([]);
  };

  if (filters.length === 0) return null;

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <FiFilter size={16} className="text-gray-500 dark:text-gray-400" />
      {visibleFilters.map((filter) => (
        <motion.button
          key={filter.id}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => toggleFilter(filter.id)}
          className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
            selected.includes(filter.id)
              ? 'bg-primary-600 text-white'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
          }`}
        >
          {filter.label}
          {filter.count !== undefined && (
            <span className="ml-1 opacity-75">({filter.count})</span>
          )}
        </motion.button>
      ))}
      {filters.length > maxVisible && (
        <button
          onClick={() => setShowAll(!showAll)}
          className="text-sm text-primary-600 dark:text-primary-400 hover:underline"
        >
          {showAll ? 'Mostrar menos' : `+${filters.length - maxVisible} más`}
        </button>
      )}
      {selected.length > 0 && (
        <button
          onClick={clearAll}
          className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
        >
          <FiX size={14} />
          Limpiar
        </button>
      )}
    </div>
  );
}

