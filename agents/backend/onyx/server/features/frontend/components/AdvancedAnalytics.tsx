'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiTrendingUp, FiTrendingDown, FiBarChart2, FiPieChart } from 'react-icons/fi';
import AnalyticsChart from './AnalyticsChart';

interface AdvancedAnalyticsProps {
  taskId?: string;
}

export default function AdvancedAnalytics({ taskId }: AdvancedAnalyticsProps) {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d');
  const [metrics, setMetrics] = useState({
    totalDocuments: 0,
    avgGenerationTime: 0,
    successRate: 0,
    mostUsedTemplate: '',
    peakHours: [] as number[],
  });

  useEffect(() => {
    // Load analytics from localStorage
    const stored = localStorage.getItem('bul_analytics');
    if (stored) {
      const data = JSON.parse(stored);
      setMetrics({
        totalDocuments: data.totalDocuments || 0,
        avgGenerationTime: data.avgGenerationTime || 0,
        successRate: data.successRate || 0,
        mostUsedTemplate: data.mostUsedTemplate || 'N/A',
        peakHours: data.peakHours || [],
      });
    }
  }, []);

  const chartData = [
    { label: 'Lun', value: 12, color: 'bg-blue-500' },
    { label: 'Mar', value: 19, color: 'bg-blue-500' },
    { label: 'Mié', value: 15, color: 'bg-blue-500' },
    { label: 'Jue', value: 22, color: 'bg-blue-500' },
    { label: 'Vie', value: 18, color: 'bg-blue-500' },
    { label: 'Sáb', value: 8, color: 'bg-blue-500' },
    { label: 'Dom', value: 5, color: 'bg-blue-500' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Analytics Avanzados
        </h3>
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value as any)}
          className="select text-sm"
        >
          <option value="7d">Últimos 7 días</option>
          <option value="30d">Últimos 30 días</option>
          <option value="90d">Últimos 90 días</option>
          <option value="all">Todo el tiempo</option>
        </select>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Total Documentos</span>
            <FiBarChart2 size={18} className="text-primary-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {metrics.totalDocuments}
          </div>
          <div className="flex items-center gap-1 mt-2 text-xs text-green-600">
            <FiTrendingUp size={14} />
            <span>+12%</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Tiempo Promedio</span>
            <FiPieChart size={18} className="text-primary-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {metrics.avgGenerationTime}s
          </div>
          <div className="flex items-center gap-1 mt-2 text-xs text-red-600">
            <FiTrendingDown size={14} />
            <span>-5%</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Tasa de Éxito</span>
            <FiTrendingUp size={18} className="text-primary-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {metrics.successRate.toFixed(1)}%
          </div>
          <div className="flex items-center gap-1 mt-2 text-xs text-green-600">
            <FiTrendingUp size={14} />
            <span>+2.3%</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Plantilla Más Usada</span>
            <FiBarChart2 size={18} className="text-primary-600" />
          </div>
          <div className="text-lg font-bold text-gray-900 dark:text-white truncate">
            {metrics.mostUsedTemplate}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            Últimos 30 días
          </div>
        </motion.div>
      </div>

      <AnalyticsChart
        title="Documentos Generados por Día"
        data={chartData}
        type="bar"
      />
    </div>
  );
}


