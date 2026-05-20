'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/lib/api-client';
import { useAppStore } from '@/store/app-store';
import type { StatsResponse, TaskListItem } from '@/types/api';
import { motion } from 'framer-motion';
import { FiTrendingUp, FiClock, FiCheckCircle, FiXCircle, FiActivity } from 'react-icons/fi';
import ActivityFeed from './ActivityFeed';
import QuickStats from './QuickStats';
import TimeTracker from './TimeTracker';

export default function Dashboard() {
  const { stats } = useAppStore();
  const [recentTasks, setRecentTasks] = useState<TaskListItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [tasksResponse] = await Promise.all([
          apiClient.listTasks({ limit: 5, offset: 0 }),
        ]);
        setRecentTasks(tasksResponse.tasks);
      } catch (error) {
        console.error('Error loading dashboard data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, []);

  const displayStats = stats;

  const statCards = [
    {
      icon: FiActivity,
      label: 'Total Solicitudes',
      value: displayStats?.total_requests.toLocaleString() || '0',
      color: 'bg-blue-500',
      trend: '+12%',
    },
    {
      icon: FiClock,
      label: 'Tareas Activas',
      value: displayStats?.active_tasks.toString() || '0',
      color: 'bg-yellow-500',
      trend: null,
    },
    {
      icon: FiCheckCircle,
      label: 'Completadas',
      value: displayStats?.completed_tasks.toLocaleString() || '0',
      color: 'bg-green-500',
      trend: '+5%',
    },
    {
      icon: FiTrendingUp,
      label: 'Tasa de Éxito',
      value: displayStats ? `${(displayStats.success_rate * 100).toFixed(1)}%` : '0%',
      color: 'bg-purple-500',
      trend: '+2.3%',
    },
  ];

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Dashboard</h2>
        <p className="text-gray-600 dark:text-gray-400">Vista general del sistema BUL</p>
      </motion.div>

      {/* Quick Stats */}
      <QuickStats />

      {/* Stats Grid with TimeTracker */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
        {statCards.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="card hover:shadow-lg transition-shadow"
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`${stat.color} p-3 rounded-lg`}>
                <stat.icon className="text-white" size={24} />
              </div>
              {stat.trend && (
                <span className="text-sm text-green-600 font-medium">{stat.trend}</span>
              )}
            </div>
            <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
              {stat.value}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</div>
          </motion.div>
        ))}
      </div>

      {/* Recent Tasks */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Tareas Recientes</h3>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {recentTasks.length} tareas
          </span>
        </div>
        
        {isLoading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
          </div>
        ) : recentTasks.length === 0 ? (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No hay tareas recientes
          </div>
        ) : (
          <div className="space-y-3">
            {recentTasks.map((task, index) => (
              <motion.div
                key={task.task_id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
                className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {task.query_preview}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 font-mono">
                    {task.task_id}
                  </p>
                </div>
                <div className="ml-4 flex items-center gap-2">
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      task.status === 'completed'
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : task.status === 'processing'
                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                        : task.status === 'failed'
                        ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                        : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                    }`}
                  >
                    {task.status}
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>

      {/* TimeTracker and Activity Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="lg:col-span-1"
        >
          <TimeTracker />
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="lg:col-span-2"
        >
          <ActivityFeed />
        </motion.div>
      </div>

      {/* Performance Chart Placeholder */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="card"
      >
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Rendimiento del Sistema
        </h3>
        <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
          <p className="text-gray-500 dark:text-gray-400">
            Gráfico de rendimiento (próximamente)
          </p>
        </div>
      </motion.div>
    </div>
  );
}

