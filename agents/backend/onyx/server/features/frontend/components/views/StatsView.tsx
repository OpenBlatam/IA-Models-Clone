'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/lib/api-client';
import { useAppStore } from '@/store/app-store';
import type { StatsResponse } from '@/types/api';
import AnalyticsChart from '@/components/AnalyticsChart';
import AdvancedAnalytics from '@/components/AdvancedAnalytics';

export default function StatsView() {
  const { stats } = useAppStore();
  const [localStats, setLocalStats] = useState<StatsResponse | null>(null);

  useEffect(() => {
    const loadStats = async () => {
      try {
        const statsData = await apiClient.getStats();
        setLocalStats(statsData);
      } catch (error) {
        console.error('Failed to load stats:', error);
      }
    };

    loadStats();
    const interval = setInterval(loadStats, 10000);
    return () => clearInterval(interval);
  }, []);

  const displayStats = localStats || stats;

  const formatUptime = (uptime: string) => {
    try {
      // Parse uptime string (format: "X days, HH:MM:SS")
      const parts = uptime.split(', ');
      if (parts.length === 2) {
        const days = parts[0].split(' ')[0];
        const time = parts[1];
        return `${days} días, ${time}`;
      }
      return uptime;
    } catch {
      return uptime;
    }
  };

  const statCards = [
    {
      icon: '📊',
      label: 'Total Solicitudes',
      value: displayStats?.total_requests.toLocaleString() || '-',
      color: 'bg-blue-500',
    },
    {
      icon: '⚡',
      label: 'Tareas Activas',
      value: displayStats?.active_tasks.toString() || '-',
      color: 'bg-yellow-500',
    },
    {
      icon: '✅',
      label: 'Tareas Completadas',
      value: displayStats?.completed_tasks.toLocaleString() || '-',
      color: 'bg-green-500',
    },
    {
      icon: '📈',
      label: 'Tasa de Éxito',
      value: displayStats ? `${(displayStats.success_rate * 100).toFixed(1)}%` : '-',
      color: 'bg-purple-500',
    },
    {
      icon: '⏱️',
      label: 'Tiempo Promedio',
      value: displayStats ? `${displayStats.average_processing_time.toFixed(2)}s` : '-',
      color: 'bg-indigo-500',
    },
    {
      icon: '🕐',
      label: 'Tiempo Activo',
      value: displayStats ? formatUptime(displayStats.uptime) : '-',
      color: 'bg-pink-500',
    },
  ];

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Estadísticas del Sistema</h2>
        <p className="text-gray-600">Métricas en tiempo real del sistema BUL</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {statCards.map((stat, index) => (
          <div key={index} className="card">
            <div className="flex items-center gap-4">
              <div className={`${stat.color} w-12 h-12 rounded-lg flex items-center justify-center text-2xl`}>
                {stat.icon}
              </div>
              <div className="flex-1">
                <div className="text-2xl font-bold text-gray-900 mb-1">{stat.value}</div>
                <div className="text-sm text-gray-600">{stat.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {!displayStats && (
        <div className="card text-center py-12 mt-6">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Cargando estadísticas...</p>
        </div>
      )}

      {displayStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          <AnalyticsChart
            title="Tareas por Estado"
            data={[
              { label: 'Completadas', value: displayStats.completed_tasks, color: 'bg-green-500' },
              { label: 'Activas', value: displayStats.active_tasks, color: 'bg-yellow-500' },
            ]}
            type="bar"
          />
          <AnalyticsChart
            title="Rendimiento"
            data={[
              { label: 'Tasa de Éxito', value: Math.round(displayStats.success_rate * 100), color: 'bg-blue-500' },
              { label: 'Tiempo Promedio (s)', value: Math.round(displayStats.average_processing_time), color: 'bg-purple-500' },
            ]}
            type="bar"
          />
        </div>
      )}

      <div className="mt-8">
        <AdvancedAnalytics />
      </div>
    </div>
  );
}

