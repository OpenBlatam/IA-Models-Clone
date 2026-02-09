'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiFilter, FiX, FiCalendar, FiTag } from 'react-icons/fi';

interface FilterOptions {
  status?: string[];
  businessArea?: string[];
  documentType?: string[];
  dateRange?: {
    start: string;
    end: string;
  };
  priority?: number[];
}

interface AdvancedFiltersProps {
  onApply: (filters: FilterOptions) => void;
  onReset: () => void;
}

export default function AdvancedFilters({ onApply, onReset }: AdvancedFiltersProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [filters, setFilters] = useState<FilterOptions>({});

  const handleApply = () => {
    onApply(filters);
    setIsOpen(false);
  };

  const handleReset = () => {
    setFilters({});
    onReset();
    setIsOpen(false);
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="btn btn-secondary"
      >
        <FiFilter size={18} />
        Filtros Avanzados
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            <div
              className="fixed inset-0 bg-black bg-opacity-50 z-40"
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="fixed inset-x-4 top-20 bg-white dark:bg-gray-800 rounded-xl shadow-xl z-50 max-w-2xl mx-auto p-6 max-h-[80vh] overflow-y-auto"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">Filtros Avanzados</h3>
                <button onClick={() => setIsOpen(false)} className="btn-icon">
                  <FiX size={20} />
                </button>
              </div>

              <div className="space-y-6">
                {/* Status Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Estado
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {['queued', 'processing', 'completed', 'failed', 'cancelled'].map((status) => (
                      <button
                        key={status}
                        onClick={() => {
                          const current = filters.status || [];
                          const updated = current.includes(status)
                            ? current.filter((s) => s !== status)
                            : [...current, status];
                          setFilters({ ...filters, status: updated });
                        }}
                        className={`px-3 py-1 rounded-full text-sm ${
                          filters.status?.includes(status)
                            ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                            : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                        }`}
                      >
                        {status}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Business Area Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    <FiTag className="inline mr-1" />
                    Área de Negocio
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {['marketing', 'ventas', 'recursos-humanos', 'tecnologia', 'finanzas', 'operaciones'].map((area) => (
                      <button
                        key={area}
                        onClick={() => {
                          const current = filters.businessArea || [];
                          const updated = current.includes(area)
                            ? current.filter((a) => a !== area)
                            : [...current, area];
                          setFilters({ ...filters, businessArea: updated });
                        }}
                        className={`px-3 py-1 rounded-full text-sm ${
                          filters.businessArea?.includes(area)
                            ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                            : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                        }`}
                      >
                        {area}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Date Range */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    <FiCalendar className="inline mr-1" />
                    Rango de Fechas
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <input
                        type="date"
                        value={filters.dateRange?.start || ''}
                        onChange={(e) =>
                          setFilters({
                            ...filters,
                            dateRange: { ...filters.dateRange, start: e.target.value } as any,
                          })
                        }
                        className="input"
                      />
                    </div>
                    <div>
                      <input
                        type="date"
                        value={filters.dateRange?.end || ''}
                        onChange={(e) =>
                          setFilters({
                            ...filters,
                            dateRange: { ...filters.dateRange, end: e.target.value } as any,
                          })
                        }
                        className="input"
                      />
                    </div>
                  </div>
                </div>

                {/* Priority Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Prioridad
                  </label>
                  <div className="flex gap-2">
                    {[1, 2, 3, 4, 5].map((priority) => (
                      <button
                        key={priority}
                        onClick={() => {
                          const current = filters.priority || [];
                          const updated = current.includes(priority)
                            ? current.filter((p) => p !== priority)
                            : [...current, priority];
                          setFilters({ ...filters, priority: updated });
                        }}
                        className={`px-4 py-2 rounded-lg text-sm ${
                          filters.priority?.includes(priority)
                            ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                            : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                        }`}
                      >
                        {priority}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-end gap-3 mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <button onClick={handleReset} className="btn btn-secondary">
                  Limpiar
                </button>
                <button onClick={handleApply} className="btn btn-primary">
                  Aplicar Filtros
                </button>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}


