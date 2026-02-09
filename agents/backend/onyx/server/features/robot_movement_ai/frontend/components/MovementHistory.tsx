'use client';

import { useEffect } from 'react';
import { FixedSizeList as List } from 'react-window';
import { apiClient } from '@/lib/api/client';
import { useAsync } from '@/lib/hooks/useAsync';
import { useFilter } from '@/lib/hooks/useFilter';
import { useSort } from '@/lib/hooks/useSort';
import { Clock, MapPin, CheckCircle, XCircle } from 'lucide-react';
import { format } from 'date-fns';
import { motion } from 'framer-motion';
import { logger } from '@/lib/utils/logger';

interface Movement {
  timestamp: string;
  target: { x: number; y: number; z: number };
  success: boolean;
  duration?: number;
}

export default function MovementHistory() {
  const { execute: loadHistory, data: historyData, loading: isLoading } = useAsync(
    async () => {
      const data = await apiClient.getMovementHistory(20);
      return data.history || [];
    }
  );

  const history = historyData || [];

  // Filter and sort
  const { filter, setFilter, filteredData } = useFilter(history, {
    searchFields: ['timestamp'],
  });

  const { sortedData, toggleSort, sortKey, sortDirection } = useSort(filteredData, {
    initialSortKey: 'timestamp',
    initialSortDirection: 'desc',
  });

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  return (
    <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
      <div className="flex items-center justify-between mb-tesla-lg">
        <h3 className="text-lg font-semibold text-tesla-black">Historial de Movimientos</h3>
        <button
          onClick={loadHistory}
          disabled={isLoading}
          className="text-sm text-tesla-blue hover:text-opacity-80 font-medium disabled:opacity-50 transition-colors"
        >
          Actualizar
        </button>
      </div>

      {/* Filter input */}
      <div className="mb-tesla-md">
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Buscar por fecha..."
          className="w-full px-tesla-md py-tesla-sm bg-white border border-gray-300 rounded-md text-tesla-black placeholder-tesla-gray-light focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all"
        />
      </div>

      {isLoading ? (
        <div className="text-center text-tesla-gray-dark py-tesla-xl">Cargando...</div>
      ) : sortedData.length === 0 ? (
        <div className="text-center text-tesla-gray-dark py-tesla-xl">No hay movimientos registrados</div>
      ) : (
        <div className="max-h-[400px]">
          <List
            height={400}
            itemCount={sortedData.length}
            itemSize={100}
            width="100%"
            className="scrollbar-hide"
          >
            {({ index, style }) => {
              const movement = sortedData[index];
              return (
                <div style={style} className="px-tesla-sm">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2, delay: index * 0.05 }}
                    className="p-tesla-md bg-gray-50 rounded-md border border-gray-200 hover:border-gray-300 hover:shadow-sm transition-all mb-tesla-sm"
                  >
                    <div className="flex items-start justify-between gap-tesla-md">
                      <div className="flex-1">
                        <div className="flex items-center gap-tesla-sm mb-tesla-sm">
                          {movement.success ? (
                            <CheckCircle className="w-4 h-4 text-green-600" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-600" />
                          )}
                          <span className="text-sm text-tesla-gray-dark font-medium">
                            {movement.timestamp
                              ? format(new Date(movement.timestamp), 'HH:mm:ss')
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="flex items-center gap-tesla-sm text-sm">
                          <MapPin className="w-4 h-4 text-tesla-blue" />
                          <span className="text-tesla-black font-mono">
                            ({movement.target?.x?.toFixed(3) || 0},{' '}
                            {movement.target?.y?.toFixed(3) || 0},{' '}
                            {movement.target?.z?.toFixed(3) || 0})
                          </span>
                        </div>
                        {movement.duration && (
                          <div className="flex items-center gap-tesla-sm mt-tesla-sm text-xs text-tesla-gray-dark">
                            <Clock className="w-3 h-3" />
                            <span>{movement.duration.toFixed(2)}s</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                </div>
              );
            }}
          </List>
        </div>
      )}
    </div>
  );
}

