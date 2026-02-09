'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiSearch, FiX, FiFilter, FiCalendar, FiTag } from 'react-icons/fi';
import { format } from 'date-fns';

interface SearchFilter {
  query?: string;
  status?: string[];
  dateRange?: {
    start: Date | null;
    end: Date | null;
  };
  businessArea?: string[];
  tags?: string[];
}

interface AdvancedSearchProps {
  onSearch: (filters: SearchFilter) => void;
  onReset: () => void;
  initialFilters?: SearchFilter;
}

export default function AdvancedSearch({
  onSearch,
  onReset,
  initialFilters = {},
}: AdvancedSearchProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [filters, setFilters] = useState<SearchFilter>(initialFilters);

  const handleApply = () => {
    onSearch(filters);
    setIsOpen(false);
  };

  const handleReset = () => {
    setFilters({});
    onReset();
    setIsOpen(false);
  };

  const hasActiveFilters = Object.keys(filters).some(
    (key) => {
      const value = filters[key as keyof SearchFilter];
      if (Array.isArray(value)) return value.length > 0;
      if (value && typeof value === 'object' && 'start' in value) {
        return value.start !== null || value.end !== null;
      }
      return !!value;
    }
  );

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className={`btn btn-secondary relative ${hasActiveFilters ? 'bg-primary-50 dark:bg-primary-900/30' : ''}`}
        title="Búsqueda avanzada"
      >
        <FiSearch size={18} />
        {hasActiveFilters && (
          <span className="absolute -top-1 -right-1 w-3 h-3 bg-primary-600 rounded-full"></span>
        )}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  Búsqueda Avanzada
                </h3>
                <button onClick={() => setIsOpen(false)} className="btn-icon">
                  <FiX size={20} />
                </button>
              </div>

              <div className="p-6 space-y-6">
                {/* Query Search */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    <FiSearch size={16} className="inline mr-2" />
                    Buscar en texto
                  </label>
                  <input
                    type="text"
                    value={filters.query || ''}
                    onChange={(e) => setFilters({ ...filters, query: e.target.value })}
                    placeholder="Buscar en consultas, IDs..."
                    className="input w-full"
                  />
                </div>

                {/* Status Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    <FiFilter size={16} className="inline mr-2" />
                    Estado
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {['queued', 'processing', 'completed', 'failed'].map((status) => (
                      <button
                        key={status}
                        onClick={() => {
                          const current = filters.status || [];
                          setFilters({
                            ...filters,
                            status: current.includes(status)
                              ? current.filter((s) => s !== status)
                              : [...current, status],
                          });
                        }}
                        className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                          filters.status?.includes(status)
                            ? 'bg-primary-600 text-white'
                            : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                        }`}
                      >
                        {status}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Date Range */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    <FiCalendar size={16} className="inline mr-2" />
                    Rango de Fechas
                  </label>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">
                        Desde
                      </label>
                      <input
                        type="date"
                        value={
                          filters.dateRange?.start
                            ? format(filters.dateRange.start, 'yyyy-MM-dd')
                            : ''
                        }
                        onChange={(e) =>
                          setFilters({
                            ...filters,
                            dateRange: {
                              ...filters.dateRange,
                              start: e.target.value ? new Date(e.target.value) : null,
                              end: filters.dateRange?.end || null,
                            },
                          })
                        }
                        className="input w-full"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">
                        Hasta
                      </label>
                      <input
                        type="date"
                        value={
                          filters.dateRange?.end
                            ? format(filters.dateRange.end, 'yyyy-MM-dd')
                            : ''
                        }
                        onChange={(e) =>
                          setFilters({
                            ...filters,
                            dateRange: {
                              ...filters.dateRange,
                              start: filters.dateRange?.start || null,
                              end: e.target.value ? new Date(e.target.value) : null,
                            },
                          })
                        }
                        className="input w-full"
                      />
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <button onClick={handleApply} className="btn btn-primary flex-1">
                    Aplicar Filtros
                  </button>
                  <button onClick={handleReset} className="btn btn-secondary">
                    Limpiar
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

