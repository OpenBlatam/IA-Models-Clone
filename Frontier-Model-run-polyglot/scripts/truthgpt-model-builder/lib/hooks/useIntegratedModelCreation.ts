/**
 * Hook integrado completo para creación de modelos
 * =================================================
 * 
 * Combina todos los hooks y utilidades en una solución completa
 */

import { useCallback, useMemo } from 'react'
import { useSmartModelCreation } from './useSmartModelCreation'
import { useModelNotifications } from './useModelNotifications'
import { useModelCache } from './useModelCache'
import { useModelQueue } from './useModelQueue'
import { useDebouncedModelCreation } from './useDebouncedModelCreation'
import { TruthGPTAPIClient } from '../truthgpt-api-client'
import { mergeModelSpecs, generateModelNameFromDescription } from '../modelCreationHelpers'

export interface IntegratedModelCreationOptions {
  description: string
  spec?: any
  useQueue?: boolean
  priority?: number
  enableCache?: boolean
  enableValidation?: boolean
  enableAnalysis?: boolean
  showNotifications?: boolean
}

export interface UseIntegratedModelCreationResult {
  createModel: (options: IntegratedModelCreationOptions) => Promise<string | null>
  validateDescription: (description: string) => void
  isCreating: boolean
  isValidationPending: boolean
  validationResult: { isValid: boolean; errors: string[] } | null
  activeModels: Set<string>
  queueStats: {
    total: number
    pending: number
    processing: number
    completed: number
    failed: number
  }
  cacheStats: {
    size: number
    hitRate: number
  }
  clearCache: () => void
  clearQueue: () => void
}

/**
 * Hook completo que integra todas las funcionalidades
 */
export function useIntegratedModelCreation(
  apiClient: TruthGPTAPIClient | null,
  apiConnected: boolean
): UseIntegratedModelCreationResult {
  // Hook principal de creación
  const smartCreation = useSmartModelCreation(apiClient, apiConnected)

  // Notificaciones
  const notifications = useModelNotifications({
    showProgress: true,
    showSuccess: true,
    showErrors: true,
    showWarnings: true
  })

  // Caché de modelos
  const modelCache = useModelCache({
    maxSize: 50,
    ttl: 24 * 60 * 60 * 1000, // 24 horas
    enableAutoCleanup: true
  })

  // Validación con debounce
  const debouncedValidation = useDebouncedModelCreation({
    debounceDelay: 500,
    onValidationChange: (isValid, errors) => {
      if (!isValid && errors.length > 0) {
        notifications.notifyWarning(errors[0])
      }
    }
  })

  // Wrapper para creación que usa caché
  const createModelWithCache = useCallback(
    async (description: string, spec?: any): Promise<string | null> => {
      // Verificar caché
      const cached = modelCache.get(description)
      if (cached) {
        notifications.notifyInfo('Modelo encontrado en caché')
        return cached.id
      }

      // Crear modelo
      const modelId = await smartCreation.createModel({
        description,
        spec,
        enableCache: true,
        enableValidation: true,
        enableAnalysis: true,
        onSuccess: (id, name) => {
          notifications.notifyModelCreated(id, name)
          // Guardar en caché
          modelCache.set(description, {
            id,
            name,
            description,
            spec: spec || null
          })
        },
        onError: (error) => {
          notifications.notifyError('', error)
        }
      })

      return modelId
    },
    [smartCreation, modelCache, notifications]
  )

  // Cola de modelos
  const modelQueue = useModelQueue(
    createModelWithCache,
    {
      maxConcurrent: 1,
      autoProcess: true
    }
  )

  // Función principal de creación
  const createModel = useCallback(
    async (options: IntegratedModelCreationOptions): Promise<string | null> => {
      const {
        description,
        spec,
        useQueue = false,
        priority = 0,
        enableCache = true,
        enableValidation = true,
        enableAnalysis = true
      } = options

      // Validar descripción si está habilitado
      if (enableValidation) {
        debouncedValidation.validateDescription(description)
        const validation = debouncedValidation.validationResult
        if (validation && !validation.isValid) {
          return null
        }
      }

      // Preparar spec
      const finalSpec = spec ? {
        ...spec,
        modelName: spec.modelName || generateModelNameFromDescription(description)
      } : {
        modelName: generateModelNameFromDescription(description)
      }

      // Usar cola si está habilitado
      if (useQueue) {
        const queueId = modelQueue.enqueue(description, finalSpec, priority)
        notifications.notifyInfo(`Modelo agregado a la cola (prioridad: ${priority})`)
        return queueId
      }

      // Crear directamente
      return await createModelWithCache(description, finalSpec)
    },
    [createModelWithCache, modelQueue, debouncedValidation, notifications]
  )

  return {
    createModel,
    validateDescription: debouncedValidation.validateDescription,
    isCreating: smartCreation.isCreating || modelQueue.stats.processing > 0,
    isValidationPending: debouncedValidation.isValidationPending,
    validationResult: debouncedValidation.validationResult,
    activeModels: smartCreation.activeModels,
    queueStats: modelQueue.stats,
    cacheStats: {
      size: modelCache.size,
      hitRate: modelCache.stats.hitRate
    },
    clearCache: () => {
      modelCache.clear()
      smartCreation.clearCache()
    },
    clearQueue: modelQueue.clear
  }
}

