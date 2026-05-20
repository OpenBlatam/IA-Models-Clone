/**
 * Hook inteligente para creación de modelos con todas las optimizaciones
 * ======================================================================
 * 
 * Combina todas las mejoras: validación, análisis, creación, monitoreo y métricas
 */

import { useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { useModelOperations } from './useModelOperations'
import { useOptimizedModelCreation } from './useOptimizedModelCreation'
import { useModelPerformance } from './useModelPerformance'
import { TruthGPTAPIClient } from '../truthgpt-api-client'

export interface SmartModelCreationOptions {
  description: string
  spec?: any
  enableValidation?: boolean
  enableAnalysis?: boolean
  enableCache?: boolean
  showWarnings?: boolean
  onSuccess?: (modelId: string, modelName: string) => void
  onError?: (error: Error) => void
}

export interface UseSmartModelCreationResult {
  createModel: (options: SmartModelCreationOptions) => Promise<string | null>
  isCreating: boolean
  activeModels: Set<string>
  error: string | null
  performance: ReturnType<typeof useModelPerformance>
  clearCache: () => void
}

/**
 * Hook que combina todas las optimizaciones para creación de modelos
 */
export function useSmartModelCreation(
  apiClient: TruthGPTAPIClient | null,
  apiConnected: boolean
): UseSmartModelCreationResult {
  const {
    createAndMonitorModel,
    isCreating,
    activeModels,
    error
  } = useModelOperations(apiClient, apiConnected)

  const {
    prepareModelCreation,
    clearCache
  } = useOptimizedModelCreation()

  const performance = useModelPerformance()

  const createModel = useCallback(
    async (options: SmartModelCreationOptions): Promise<string | null> => {
      const {
        description,
        spec,
        enableValidation = true,
        enableAnalysis = true,
        enableCache = true,
        showWarnings = true,
        onSuccess,
        onError
      } = options

      // Iniciar timer de rendimiento
      const stopTimer = performance.startTimer('creation')

      try {
        // Preparar y validar descripción
        if (enableValidation || enableAnalysis) {
          const prepared = await prepareModelCreation({
            description,
            enableCache,
            enableValidation,
            enableAnalysis
          })

          // Mostrar errores de validación
          if (!prepared.isValid) {
            const errorMessage = prepared.errors.join(', ')
            toast.error(errorMessage, { duration: 5000 })
            onError?.(new Error(errorMessage))
            return null
          }

          // Mostrar advertencias si las hay
          if (showWarnings && prepared.warnings.length > 0) {
            prepared.warnings.forEach(warning => {
              toast(warning, { icon: '⚠️', duration: 3000 })
            })
          }

          // Usar análisis si está disponible
          if (prepared.analysis && spec) {
            spec.suggestedLayers = prepared.analysis.suggestedLayers
            spec.estimatedComplexity = prepared.analysis.estimatedComplexity
            spec.recommendedOptimizer = prepared.analysis.recommendedOptimizer
            spec.recommendedLoss = prepared.analysis.recommendedLoss
          }
        }

        // Registrar llamada API
        performance.recordApiCall(true)

        // Crear y monitorear modelo
        const modelId = await createAndMonitorModel(
          description,
          spec,
          {
            onModelCreated: (id, name) => {
              stopTimer()
              performance.recordApiCall(true)
              toast.success(`Modelo ${name} creado`, { 
                icon: '✅',
                duration: 3000 
              })
              onSuccess?.(id, name)
            },
            onStatusUpdate: (id, status) => {
              // Actualizaciones silenciosas durante el polling
              if (status.progress && status.progress % 25 === 0) {
                toast(`Progreso: ${status.progress}%`, {
                  icon: '⏳',
                  duration: 2000
                })
              }
            },
            onModelCompleted: (id, githubUrl) => {
              performance.recordApiCall(true)
              const githubUrlStr = githubUrl || 'GitHub'
              toast.success(`¡Modelo completado! ${githubUrlStr}`, {
                icon: '🎉',
                duration: 5000
              })
            },
            onModelFailed: (id, error) => {
              stopTimer()
              performance.recordApiCall(false)
              const errorMessage = error.message || 'Error desconocido'
              toast.error(`Error: ${errorMessage}`, {
                icon: '❌',
                duration: 5000
              })
              onError?.(error)
            }
          }
        )

        return modelId
      } catch (error) {
        stopTimer()
        performance.recordApiCall(false)
        const errorMessage = error instanceof Error ? error.message : 'Error desconocido'
        toast.error(`Error al crear modelo: ${errorMessage}`, {
          icon: '❌',
          duration: 5000
        })
        onError?.(error instanceof Error ? error : new Error(errorMessage))
        return null
      }
    },
    [createAndMonitorModel, prepareModelCreation, performance]
  )

  return {
    createModel,
    isCreating,
    activeModels,
    error,
    performance,
    clearCache
  }
}

