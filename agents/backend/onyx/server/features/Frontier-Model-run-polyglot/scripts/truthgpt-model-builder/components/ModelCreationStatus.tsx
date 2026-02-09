/**
 * Componente Mejorado para mostrar el estado de creación de modelos
 * ===================================================================
 * 
 * Versión mejorada con tipos estrictos y mejor feedback visual
 */

'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, XCircle, Loader2, Clock, Activity, TrendingUp, Zap } from 'lucide-react'
import { 
  StatusBadge, 
  ProgressBar, 
  LoadingSpinner,
  ModelProgressCard 
} from './LoadingStates'
import { 
  ModelStatus, 
  QueueStats, 
  CacheStats, 
  ValidationResult,
  ModelProgress 
} from '@/lib/types/modelTypes'

export interface ModelCreationStatusProps {
  isCreating: boolean
  activeModels: Set<string>
  queueStats?: QueueStats
  cacheStats?: CacheStats
  validationPending?: boolean
  validationResult?: ValidationResult | null
  currentProgress?: ModelProgress
  onCancel?: (modelId: string) => void
  className?: string
}

export default function ModelCreationStatus({
  isCreating,
  activeModels,
  queueStats,
  cacheStats,
  validationPending,
  validationResult,
  currentProgress,
  onCancel,
  className = ''
}: ModelCreationStatusProps) {
  const hasActiveModels = activeModels.size > 0
  const hasQueue = queueStats && queueStats.total > 0

  if (!isCreating && !hasActiveModels && !hasQueue && !validationPending && !currentProgress) {
    return null
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className={`fixed top-4 right-4 z-50 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 p-4 min-w-[320px] max-w-[420px] ${className}`}
      >
        <div className="space-y-4">
          {/* Título */}
          <div className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 pb-2">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Estado del Sistema
            </h3>
          </div>

          {/* Progreso actual detallado */}
          {currentProgress && (
            <ModelProgressCard
              progress={currentProgress}
              onCancel={onCancel ? () => onCancel(currentProgress.modelId) : undefined}
            />
          )}

          {/* Estado de validación mejorado */}
          {validationPending && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg"
            >
              <LoadingSpinner size="sm" variant="pulse" color="primary" />
              <div className="flex-1">
                <p className="text-sm font-medium text-blue-900 dark:text-blue-100">
                  Validando descripción
                </p>
                <p className="text-xs text-blue-700 dark:text-blue-300">
                  Verificando especificaciones...
                </p>
              </div>
            </motion.div>
          )}

          {validationResult && !validationResult.valid && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800"
            >
              <div className="flex items-start gap-2">
                <XCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-red-900 dark:text-red-100 mb-1">
                    Errores de validación
                  </p>
                  <ul className="text-xs text-red-700 dark:text-red-300 space-y-1">
                    {validationResult.errors.map((error, idx) => (
                      <li key={idx} className="flex items-start gap-1">
                        <span className="text-red-500">•</span>
                        <span>{error.message}</span>
                      </li>
                    ))}
                  </ul>
                  {validationResult.warnings.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-red-200 dark:border-red-800">
                      <p className="text-xs font-medium text-red-800 dark:text-red-200 mb-1">
                        Advertencias:
                      </p>
                      <ul className="text-xs text-red-600 dark:text-red-400 space-y-1">
                        {validationResult.warnings.map((warning, idx) => (
                          <li key={idx} className="flex items-start gap-1">
                            <span className="text-yellow-500">⚠</span>
                            <span>{warning.message}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {/* Estado de creación mejorado */}
          {isCreating && !currentProgress && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-3 p-2 bg-purple-50 dark:bg-purple-900/20 rounded-lg"
            >
              <LoadingSpinner size="sm" variant="pulse" color="primary" />
              <div className="flex-1">
                <p className="text-sm font-medium text-purple-900 dark:text-purple-100">
                  Creando modelo
                </p>
                <p className="text-xs text-purple-700 dark:text-purple-300">
                  Procesando solicitud...
                </p>
              </div>
            </motion.div>
          )}

          {/* Modelos activos mejorado */}
          {hasActiveModels && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-3 p-2 bg-green-50 dark:bg-green-900/20 rounded-lg"
            >
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
                <span className="text-sm font-semibold text-green-900 dark:text-green-100">
                  {activeModels.size}
                </span>
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-green-900 dark:text-green-100">
                  {activeModels.size === 1 ? 'Modelo activo' : 'Modelos activos'}
                </p>
                <p className="text-xs text-green-700 dark:text-green-300">
                  En proceso de creación
                </p>
              </div>
            </motion.div>
          )}

          {/* Estadísticas de cola mejoradas */}
          {hasQueue && queueStats && (
            <div className="space-y-2 border-t border-gray-200 dark:border-gray-700 pt-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-gray-700 dark:text-gray-300">
                  Cola de Procesamiento
                </span>
                <span className="text-xs font-bold text-gray-900 dark:text-gray-100">
                  {queueStats.total}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center justify-between p-1.5 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                  <span className="text-yellow-700 dark:text-yellow-300">Pendientes:</span>
                  <span className="font-semibold text-yellow-900 dark:text-yellow-100">
                    {queueStats.pending}
                  </span>
                </div>
                <div className="flex items-center justify-between p-1.5 bg-blue-50 dark:bg-blue-900/20 rounded">
                  <span className="text-blue-700 dark:text-blue-300">Procesando:</span>
                  <span className="font-semibold text-blue-900 dark:text-blue-100">
                    {queueStats.processing}
                  </span>
                </div>
                <div className="flex items-center justify-between p-1.5 bg-green-50 dark:bg-green-900/20 rounded">
                  <span className="text-green-700 dark:text-green-300">Completados:</span>
                  <span className="font-semibold text-green-900 dark:text-green-100">
                    {queueStats.completed}
                  </span>
                </div>
                {queueStats.failed > 0 && (
                  <div className="flex items-center justify-between p-1.5 bg-red-50 dark:bg-red-900/20 rounded">
                    <span className="text-red-700 dark:text-red-300">Fallidos:</span>
                    <span className="font-semibold text-red-900 dark:text-red-100">
                      {queueStats.failed}
                    </span>
                  </div>
                )}
              </div>
              {queueStats.averageWaitTime > 0 && (
                <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
                  Tiempo promedio de espera: {queueStats.averageWaitTime.toFixed(1)}s
                </p>
              )}
            </div>
          )}

          {/* Estadísticas de caché mejoradas */}
          {cacheStats && cacheStats.size > 0 && (
            <div className="flex items-center justify-between text-xs border-t border-gray-200 dark:border-gray-700 pt-3">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                <span className="text-gray-700 dark:text-gray-300">Caché:</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-gray-600 dark:text-gray-400">
                  {cacheStats.size}/{cacheStats.maxSize}
                </span>
                <div className="flex items-center gap-1">
                  <div className="w-16 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-green-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${cacheStats.hitRate * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  <span className="text-green-600 dark:text-green-400 font-medium min-w-[3rem] text-right">
                    {(cacheStats.hitRate * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}




