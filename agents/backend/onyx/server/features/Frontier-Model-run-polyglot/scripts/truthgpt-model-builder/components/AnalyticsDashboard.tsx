/**
 * Dashboard de analytics para modelos
 * ===================================
 */

'use client'

import { motion } from 'framer-motion'
import { BarChart3, TrendingUp, AlertCircle, CheckCircle, XCircle, Clock } from 'lucide-react'

export interface AnalyticsDashboardProps {
  analytics: {
    totalCreated: number
    totalCompleted: number
    totalFailed: number
    averageCreationTime: number
    successRate: number
    errorRate: number
    mostCommonErrors: Map<string, number>
  }
  stats: {
    successRate: number
    averageTime: number
    totalOperations: number
    errorBreakdown: Array<{ error: string; count: number }>
  }
}

export default function AnalyticsDashboard({
  analytics,
  stats
}: AnalyticsDashboardProps) {
  const formatTime = (ms: number) => {
    if (ms < 1000) return `${Math.round(ms)}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    return `${(ms / 60000).toFixed(1)}min`
  }

  const formatPercentage = (value: number) => `${value.toFixed(1)}%`

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {/* Métricas principales */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Modelos Creados
          </h3>
          <BarChart3 className="w-5 h-5 text-blue-500" />
        </div>
        <div className="text-3xl font-bold text-gray-900 dark:text-white">
          {analytics.totalCreated}
        </div>
        <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          {analytics.totalCompleted} completados • {analytics.totalFailed} fallidos
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Tasa de Éxito
          </h3>
          <TrendingUp className="w-5 h-5 text-green-500" />
        </div>
        <div className="text-3xl font-bold text-green-600 dark:text-green-400">
          {formatPercentage(stats.successRate)}
        </div>
        <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          {analytics.successRate > 80 ? 'Excelente' : 
           analytics.successRate > 60 ? 'Bueno' : 
           'Necesita mejora'}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Tiempo Promedio
          </h3>
          <Clock className="w-5 h-5 text-purple-500" />
        </div>
        <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
          {formatTime(stats.averageTime)}
        </div>
        <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          Por creación de modelo
        </div>
      </motion.div>

      {/* Errores más comunes */}
      {stats.errorBreakdown.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 md:col-span-2 lg:col-span-3"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Errores Más Comunes
            </h3>
            <AlertCircle className="w-5 h-5 text-red-500" />
          </div>
          <div className="space-y-2">
            {stats.errorBreakdown.slice(0, 5).map((error, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <span className="text-sm text-gray-700 dark:text-gray-300 truncate flex-1">
                  {error.error}
                </span>
                <span className="text-sm font-semibold text-red-600 dark:text-red-400 ml-4">
                  {error.count}x
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}










