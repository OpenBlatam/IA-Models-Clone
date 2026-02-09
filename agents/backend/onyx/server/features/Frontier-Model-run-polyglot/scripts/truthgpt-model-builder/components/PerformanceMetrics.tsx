/**
 * Performance Metrics Component
 * Displays real-time performance metrics
 */

'use client'

import { useEffect, useState } from 'react'
import { Activity, TrendingUp, Clock, CheckCircle, XCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { getPerformanceDashboard } from '@/lib/service-improvements'

interface PerformanceMetricsProps {
  className?: string
  compact?: boolean
}

export default function PerformanceMetrics({ 
  className = '',
  compact = false 
}: PerformanceMetricsProps) {
  const [dashboard, setDashboard] = useState<any>(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const updateMetrics = () => {
      try {
        const data = getPerformanceDashboard()
        setDashboard(data)
      } catch (error) {
        console.error('Error fetching metrics:', error)
      }
    }

    updateMetrics()
    const interval = setInterval(updateMetrics, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [])

  if (!dashboard) return null

  const { cache, metrics, rateLimiter, health } = dashboard

  if (compact) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`flex items-center gap-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg ${className}`}
      >
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-500" />
          <span className="text-sm font-medium">Cache: {cache.messages.hitRate.toFixed(1)}%</span>
        </div>
        <div className="flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-green-500" />
          <span className="text-sm font-medium">Success: {metrics.overallSuccessRate.toFixed(1)}%</span>
        </div>
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-purple-500" />
          <span className="text-sm font-medium">Avg: {metrics.avgDuration.toFixed(0)}ms</span>
        </div>
      </motion.div>
    )
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-500" />
          Performance Metrics
        </h3>
        <button
          onClick={() => setIsVisible(!isVisible)}
          className="text-sm text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
        >
          {isVisible ? 'Hide' : 'Show Details'}
        </button>
      </div>

      <AnimatePresence>
        {isVisible && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-4"
          >
            {/* Health Status */}
            <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">System Health</span>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    health.status === 'healthy'
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      : health.status === 'degraded'
                      ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                      : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}
                >
                  {health.status.toUpperCase()}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-4 mt-2">
                <div>
                  <div className="text-xs text-gray-500 mb-1">Cache Hit Rate</div>
                  <div className="text-lg font-semibold">{health.cacheHitRate.toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Success Rate</div>
                  <div className="text-lg font-semibold">{health.successRate.toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Avg Duration</div>
                  <div className="text-lg font-semibold">{health.avgDuration.toFixed(0)}ms</div>
                </div>
              </div>
            </div>

            {/* Cache Statistics */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-blue-500" />
                  <span className="font-medium text-sm">Message Cache</span>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Hit Rate:</span>
                    <span className="font-semibold">{cache.messages.hitRate.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Size:</span>
                    <span className="font-semibold">{cache.messages.size}/{cache.messages.maxSize}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Usage:</span>
                    <span className="font-semibold">{cache.messages.usagePercent.toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-purple-500" />
                  <span className="font-medium text-sm">Model Cache</span>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Hit Rate:</span>
                    <span className="font-semibold">{cache.models.hitRate.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Size:</span>
                    <span className="font-semibold">{cache.models.size}/{cache.models.maxSize}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Usage:</span>
                    <span className="font-semibold">{cache.models.usagePercent.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Metrics Summary */}
            <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <div className="flex items-center gap-2 mb-3">
                <Clock className="w-4 h-4 text-gray-500" />
                <span className="font-medium text-sm">Performance Summary</span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-gray-500 mb-1">Total Operations</div>
                  <div className="text-lg font-semibold">{metrics.totalOperations}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Uptime</div>
                  <div className="text-lg font-semibold">
                    {Math.floor(metrics.uptimeSeconds / 60)}m {metrics.uptimeSeconds % 60}s
                  </div>
                </div>
              </div>
            </div>

            {/* Rate Limiter */}
            {rateLimiter && (
              <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-yellow-500" />
                  <span className="font-medium text-sm">Rate Limiter</span>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Current Rate:</span>
                    <span className="font-semibold">{rateLimiter.currentRate}/s</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Available:</span>
                    <span className="font-semibold">{rateLimiter.tokensAvailable.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Success Rate:</span>
                    <span className="font-semibold">{rateLimiter.successRate.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
