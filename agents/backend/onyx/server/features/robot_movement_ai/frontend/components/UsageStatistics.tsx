'use client';

import { useState, useEffect } from 'react';
import { useLocalStorage } from '@/lib/hooks/useLocalStorage';
import { BarChart3, Clock, TrendingUp, Activity } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

export default function UsageStatistics() {
  const [usageData, setUsageData] = useLocalStorage<Record<string, number>>('usage-statistics', {});
  const [totalTime, setTotalTime] = useState(0);

  useEffect(() => {
    // Calculate total time
    const total = Object.values(usageData).reduce((sum, time) => sum + time, 0);
    setTotalTime(total);
  }, [usageData]);

  const chartData = Object.entries(usageData)
    .map(([tab, time]) => ({
      name: tab.charAt(0).toUpperCase() + tab.slice(1),
      time: Math.round(time / 60), // Convert to minutes
    }))
    .sort((a, b) => b.time - a.time)
    .slice(0, 10);

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const topTabs = Object.entries(usageData)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <BarChart3 className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Estadísticas de Uso</h3>
        </div>

        {/* Summary */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-gray-400">Tiempo Total</span>
            </div>
            <p className="text-2xl font-bold text-white">{formatTime(totalTime)}</p>
          </div>
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-green-400" />
              <span className="text-sm text-gray-400">Pestañas Usadas</span>
            </div>
            <p className="text-2xl font-bold text-white">{Object.keys(usageData).length}</p>
          </div>
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-yellow-400" />
              <span className="text-sm text-gray-400">Promedio</span>
            </div>
            <p className="text-2xl font-bold text-white">
              {Object.keys(usageData).length > 0
                ? formatTime(totalTime / Object.keys(usageData).length)
                : '0m'}
            </p>
          </div>
        </div>

        {/* Top Tabs */}
        {topTabs.length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-300 mb-3">Pestañas Más Usadas</h4>
            <div className="space-y-2">
              {topTabs.map(([tab, time], index) => (
                <div key={tab} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <span className="w-6 h-6 flex items-center justify-center bg-primary-500/20 text-primary-400 rounded text-xs font-bold">
                      {index + 1}
                    </span>
                    <span className="text-white capitalize">{tab}</span>
                  </div>
                  <span className="text-gray-300 font-mono text-sm">{formatTime(time)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Chart */}
        {chartData.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-gray-300 mb-3">Distribución de Tiempo</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                />
                <Bar dataKey="time" fill="#0EA5E9" name="Minutos" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {Object.keys(usageData).length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No hay estadísticas de uso aún</p>
          </div>
        )}
      </div>
    </div>
  );
}


