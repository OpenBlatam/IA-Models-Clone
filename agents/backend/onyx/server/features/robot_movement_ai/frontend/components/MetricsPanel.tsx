'use client';

import { useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { useAsync } from '@/lib/hooks/useAsync';
import { batchAsync } from '@/lib/utils/async-helpers';
import { logger } from '@/lib/utils/logger';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { RefreshCw } from 'lucide-react';

export default function MetricsPanel() {
  const { execute: loadMetrics, data: metricsData, loading: isLoading } = useAsync(async () => {
    const results = await batchAsync([
      () => apiClient.getMetrics(),
      () => apiClient.getCPUUsage(),
      () => apiClient.getMemoryUsage(),
      () => apiClient.getPerformanceMetrics(),
    ]);
    return {
      metrics: results[0],
      cpuUsage: results[1],
      memoryUsage: results[2],
      performance: results[3],
    };
  });

  useEffect(() => {
    loadMetrics();
  }, [loadMetrics]);

  const metrics = metricsData?.metrics;
  const cpuUsage = metricsData?.cpuUsage;
  const memoryUsage = metricsData?.memoryUsage;
  const performance = metricsData?.performance;

  // Transform data for charts
  const transformMetricsForChart = (data: any) => {
    if (!data || typeof data !== 'object') return [];
    return Object.entries(data).map(([key, value]: [string, any]) => ({
      name: key,
      value: typeof value === 'object' && value.value ? value.value : value,
      timestamp: typeof value === 'object' && value.timestamp ? value.timestamp : new Date().toISOString(),
    }));
  };

  return (
    <div className="space-y-tesla-lg">
      {/* Header */}
      <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
        <div className="flex items-center justify-between mb-tesla-md">
          <h2 className="text-2xl font-semibold text-tesla-black">Métricas del Sistema</h2>
          <button
            onClick={loadMetrics}
            disabled={isLoading}
            className="flex items-center gap-tesla-sm px-tesla-md py-tesla-sm bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all disabled:opacity-50 font-medium"
          >
            <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
            Actualizar
          </button>
        </div>
      </div>

      {/* CPU Usage */}
      {cpuUsage && (
        <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
          <h3 className="text-lg font-medium text-tesla-black mb-tesla-md">Uso de CPU</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={transformMetricsForChart(cpuUsage)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" stroke="#393c41" />
              <YAxis stroke="#393c41" />
              <Tooltip
                contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px', color: '#171a20' }}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#0062cc"
                fill="#0062cc"
                fillOpacity={0.2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Memory Usage */}
      {memoryUsage && (
        <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
          <h3 className="text-lg font-medium text-tesla-black mb-4">Uso de Memoria</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={transformMetricsForChart(memoryUsage)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" stroke="#393c41" />
              <YAxis stroke="#393c41" />
              <Tooltip
                contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px', color: '#171a20' }}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Performance Metrics */}
      {performance && (
        <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
          <h3 className="text-lg font-medium text-tesla-black mb-4">Métricas de Rendimiento</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={transformMetricsForChart(performance)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" stroke="#393c41" />
              <YAxis stroke="#393c41" />
              <Tooltip
                contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px', color: '#171a20' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#0062cc"
                strokeWidth={2}
                dot={{ fill: '#0062cc' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Raw Metrics */}
      {metrics && (
        <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
          <h3 className="text-lg font-medium text-tesla-black mb-tesla-md">Todas las Métricas</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-tesla-md">
            {Object.entries(metrics).map(([key, value]: [string, any]) => (
              <div key={key} className="p-tesla-md bg-gray-50 rounded-md border border-gray-200">
                <p className="text-tesla-gray-dark text-sm mb-tesla-xs font-medium">{key}</p>
                <p className="text-tesla-black font-semibold">
                  {typeof value === 'object' && value.value
                    ? `${value.value}${value.unit || ''}`
                    : String(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {!metrics && !cpuUsage && !memoryUsage && !performance && !isLoading && (
        <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm text-center text-tesla-gray-dark">
          No hay métricas disponibles
        </div>
      )}
    </div>
  );
}

