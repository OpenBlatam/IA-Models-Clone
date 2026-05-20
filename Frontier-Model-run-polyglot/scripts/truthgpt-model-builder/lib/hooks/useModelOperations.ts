/**
 * Hook consolidado para operaciones de modelos
 * ============================================
 * 
 * Combina creación, polling y gestión de modelos en un solo hook
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { toast } from 'react-hot-toast'
import { useModelCreator, ModelCreationOptions } from './useModelCreator'
import { useModelStatusPoller } from './useModelStatusPoller'
import { TruthGPTAPIClient } from '../truthgpt-api-client'

export interface ModelOperationCallbacks {
  onModelCreated?: (modelId: string, modelName: string) => void
  onStatusUpdate?: (modelId: string, status: any) => void
  onModelCompleted?: (modelId: string, githubUrl?: string) => void
  onModelFailed?: (modelId: string, error: Error) => void
}

export interface UseModelOperationsResult {
  createAndMonitorModel: (
    description: string,
    spec?: any,
    callbacks?: ModelOperationCallbacks
  ) => Promise<string | null> // Retorna modelId o null
  cancelModelOperation: (modelId: string) => void
  isCreating: boolean
  activeModels: Set<string> // Modelos siendo monitoreados
  error: string | null
}

/**
 * Hook que combina creación y monitoreo de modelos
 */
export function useModelOperations(
  apiClient: TruthGPTAPIClient | null,
  apiConnected: boolean
): UseModelOperationsResult {
  const [isCreating, setIsCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeModels, setActiveModels] = useState<Set<string>>(new Set())
  const activeModelsRef = useRef<Set<string>>(new Set())
  const stopPollingFunctionsRef = useRef<Map<string, () => void>>(new Map())

  const { createModel: createModelWithHook } = useModelCreator(apiClient, apiConnected)
  const { startPolling, stopPolling } = useModelStatusPoller(apiClient, apiConnected)

  // Sincronizar ref con state
  useEffect(() => {
    activeModelsRef.current = activeModels
  }, [activeModels])

  const createAndMonitorModel = useCallback(
    async (
      description: string,
      spec?: any,
      callbacks?: ModelOperationCallbacks
    ): Promise<string | null> => {
      setIsCreating(true)
      setError(null)

      try {
        const result = await createModelWithHook({
          description,
          modelName: spec?.modelName,
          spec: spec || null,
          maxRetries: 3,
          retryDelay: 1000,
          onSuccess: (modelId, modelName) => {
            callbacks?.onModelCreated?.(modelId, modelName)

            // Iniciar polling inmediatamente
            const stop = startPolling({
              modelId,
              immediate: true,
              maxAttempts: 60,
              pollInterval: 5000,
              onStatusUpdate: (status) => {
                callbacks?.onStatusUpdate?.(modelId, status)
              },
              onComplete: (status) => {
                // Remover de modelos activos
                setActiveModels(prev => {
                  const next = new Set(prev)
                  next.delete(modelId)
                  return next
                })
                stopPollingFunctionsRef.current.delete(modelId)

                callbacks?.onModelCompleted?.(modelId, status.githubUrl || undefined)
              },
              onError: (error) => {
                // Remover de modelos activos
                setActiveModels(prev => {
                  const next = new Set(prev)
                  next.delete(modelId)
                  return next
                })
                stopPollingFunctionsRef.current.delete(modelId)

                callbacks?.onModelFailed?.(modelId, error)
              },
            })

            // Guardar función de stop
            stopPollingFunctionsRef.current.set(modelId, stop)

            // Agregar a modelos activos
            setActiveModels(prev => new Set(prev).add(modelId))
          },
          onError: (error) => {
            callbacks?.onModelFailed?.('', error)
          },
        })

        return result?.modelId || null
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Error desconocido'
        setError(errorMessage)
        return null
      } finally {
        setIsCreating(false)
      }
    },
    [createModelWithHook, startPolling]
  )

  const cancelModelOperation = useCallback(
    (modelId: string) => {
      // Detener polling
      const stop = stopPollingFunctionsRef.current.get(modelId)
      if (stop) {
        stop()
        stopPollingFunctionsRef.current.delete(modelId)
      } else {
        stopPolling(modelId)
      }

      // Remover de modelos activos
      setActiveModels(prev => {
        const next = new Set(prev)
        next.delete(modelId)
        return next
      })
    },
    [stopPolling]
  )

  // Limpieza al desmontar
  useEffect(() => {
    return () => {
      // Detener todos los polling activos
      stopPollingFunctionsRef.current.forEach(stop => stop())
      stopPollingFunctionsRef.current.clear()
    }
  }, [])

  return {
    createAndMonitorModel,
    cancelModelOperation,
    isCreating,
    activeModels,
    error,
  }
}

