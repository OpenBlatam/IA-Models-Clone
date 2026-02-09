'use client';

import { motion } from 'framer-motion';
import { cn } from '../../utils/cn';

interface Stats {
  total: number;
  pending: number;
  processing: number;
  running: number;
  completed: number;
  failed: number;
  withCommits: number;
  createdToday: number;
  createdThisWeek: number;
  successRate: number;
  avgProcessingTime: number;
}

interface StatsPanelProps {
  stats: Stats;
  showStats: boolean;
}

export function StatsPanel({ stats, showStats }: StatsPanelProps) {
  if (!showStats) return null;

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      className="bg-white border-b border-gray-200 px-4 py-3"
    >
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="text-xs text-gray-600 mb-1">Total</div>
          <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
        </div>
        <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
          <div className="text-xs text-yellow-700 mb-1">Procesando</div>
          <div className="text-2xl font-bold text-yellow-800">{stats.processing + stats.running}</div>
        </div>
        <div className="bg-green-50 p-3 rounded-lg border border-green-200">
          <div className="text-xs text-green-700 mb-1">Completadas</div>
          <div className="text-2xl font-bold text-green-800">{stats.completed}</div>
        </div>
        <div className="bg-red-50 p-3 rounded-lg border border-red-200">
          <div className="text-xs text-red-700 mb-1">Fallidas</div>
          <div className="text-2xl font-bold text-red-800">{stats.failed}</div>
        </div>
        <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
          <div className="text-xs text-blue-700 mb-1">Con Commits</div>
          <div className="text-2xl font-bold text-blue-800">{stats.withCommits}</div>
        </div>
      </div>
      
      {/* Gráfico de barras de estados */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Distribución por Estado</h3>
        <div className="space-y-2">
          {[
            { label: 'Completadas', value: stats.completed, color: 'bg-green-500', total: stats.total },
            { label: 'Procesando', value: stats.processing + stats.running, color: 'bg-yellow-500', total: stats.total },
            { label: 'Fallidas', value: stats.failed, color: 'bg-red-500', total: stats.total },
            { label: 'Pendientes', value: stats.pending, color: 'bg-gray-400', total: stats.total },
          ].map((item) => {
            const percentage = stats.total > 0 ? (item.value / stats.total) * 100 : 0;
            return (
              <div key={item.label} className="flex items-center gap-3">
                <div className="w-24 text-xs text-gray-600 font-medium">{item.label}</div>
                <div className="flex-1 bg-gray-200 rounded-full h-6 relative overflow-hidden">
                  <div
                    className={cn("h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2", item.color)}
                    style={{ width: `${percentage}%` }}
                  >
                    {percentage > 10 && (
                      <span className="text-xs font-semibold text-white">{item.value}</span>
                    )}
                  </div>
                </div>
                <div className="w-12 text-xs text-gray-600 text-right">{item.value}</div>
                <div className="w-12 text-xs text-gray-500 text-right">{Math.round(percentage)}%</div>
              </div>
            );
          })}
        </div>
      </div>
      
      {/* Estadísticas adicionales */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3 pt-3 border-t border-gray-200">
        <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
          <div className="text-xs text-purple-700 mb-1">Creadas hoy</div>
          <div className="text-xl font-bold text-purple-800">{stats.createdToday}</div>
        </div>
        <div className="bg-indigo-50 p-3 rounded-lg border border-indigo-200">
          <div className="text-xs text-indigo-700 mb-1">Esta semana</div>
          <div className="text-xl font-bold text-indigo-800">{stats.createdThisWeek}</div>
        </div>
        <div className="bg-teal-50 p-3 rounded-lg border border-teal-200">
          <div className="text-xs text-teal-700 mb-1">Tasa éxito</div>
          <div className="text-xl font-bold text-teal-800">{stats.successRate}%</div>
        </div>
        <div className="bg-amber-50 p-3 rounded-lg border border-amber-200">
          <div className="text-xs text-amber-700 mb-1">Tiempo promedio</div>
          <div className="text-xl font-bold text-amber-800">{stats.avgProcessingTime}m</div>
        </div>
      </div>
    </motion.div>
  );
}

