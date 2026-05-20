/**
 * Hook con Type Safety Estricto
 * ================================
 * 
 * Hook que utiliza los tipos estrictos para garantizar type safety completo
 */

import { useCallback, useState, useMemo } from 'react'
import { useCompleteModelSystem } from './useCompleteModelSystem'
import { useTruthGPTAPI } from './useTruthGPTAPI'
import {
  ModelStatus,
  ModelSpec,
  ValidationResult,
  ModelProgress,
  ModelState,
  CreateModelWithSystemOptions
} from '../types/modelTypes'

export interface UseTypedModelSystemOptions {
  enableAnalytics?: boolean
  enableOptimization?: boolean
  enableValidation?: boolean
  enableHistory?: boolean
  enableTemplates?: boolean
  enableComparison?: boolean
  enableNotifications?: boolean
}

export interface UseTypedModelSystemResult {
  // Estado
  readonly isCreating: boolean
  readonly isValidationPending: boolean
  readonly validationResult: ValidationResult | null
  readonly activeModels: ReadonlySet<string>
  readonly currentProgress: ModelProgress | null
  
  // Acciones
  createModel: (options: CreateModelWithSystemOptions) => Promise<string | null>
  validateDescription: (description: string) => ValidationResult
  cancelModel: (modelId: string) => Promise<boolean>
  
  // Utilidades
  getModelState: (modelId: string) => ModelState | null
  getAllModelStates: () => readonly ModelState[]
  clearAll: () => void
}

/**
 * Hook que proporciona un sistema de modelos con type safety estricto
 */
export function useTypedModelSystem(
  apiUrl?: string,
  options: UseTypedModelSystemOptions = {}
): UseTypedModelSystemResult {
  const {
    client: apiClient,
    isConnected: apiConnected,
    checkConnection
  } = useTruthGPTAPI(apiUrl)

  const system = useCompleteModelSystem(apiClient, apiConnected, options)

  const [modelStates, setModelStates] = useState<Map<string, ModelState>>(new Map())
  const [currentProgress, setCurrentProgress] = useState<ModelProgress | null>(null)

  // Crear modelo con type safety
  const createModel = useCallback(
    async (options: CreateModelWithSystemOptions): Promise<string | null> => {
      // Validar entrada
      if (!options.description || options.description.trim().length === 0) {
        throw new Error('La descripción del modelo es requerida')
      }

      // Crear estado inicial
      const modelId = `model-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      const initialState: ModelState = {
        id: modelId,
        name: options.spec?.name || `Model ${modelId}`,
        status: 'validating',
        progress: 0,
        createdAt: new Date(),
        updatedAt: new Date(),
        spec: options.spec as ModelSpec | undefined,
        metadata: {
          description: options.description,
          tags: options.tags || [],
          notes: options.notes
        }
      }

      setModelStates(prev => new Map(prev).set(modelId, initialState))

      // Actualizar progreso
      setCurrentProgress({
        modelId,
        step: 'validation',
        progress: 0,
        message: 'Iniciando validación...',
        startedAt: new Date(),
        updatedAt: new Date()
      })

      try {
        // Validar
        system.validateDescription(options.description)
        
        // Actualizar estado
        setModelStates(prev => {
          const updated = new Map(prev)
          const current = updated.get(modelId)
          if (current) {
            updated.set(modelId, {
              ...current,
              status: 'creating',
              progress: 10,
              updatedAt: new Date()
            })
          }
          return updated
        })

        setCurrentProgress({
          modelId,
          step: 'model_creation',
          progress: 10,
          message: 'Creando modelo...',
          startedAt: new Date(),
          updatedAt: new Date()
        })

        // Crear modelo
        const createdModelId = await system.createModel(options)

        if (createdModelId) {
          // Actualizar estado a completado
          setModelStates(prev => {
            const updated = new Map(prev)
            const current = updated.get(modelId)
            if (current) {
              updated.set(modelId, {
                ...current,
                id: createdModelId,
                status: 'completed',
                progress: 100,
                completedAt: new Date(),
                updatedAt: new Date()
              })
            }
            return updated
          })

          setCurrentProgress(null)
          return createdModelId
        }

        // Si falla, actualizar estado
        setModelStates(prev => {
          const updated = new Map(prev)
          const current = updated.get(modelId)
          if (current) {
            updated.set(modelId, {
              ...current,
              status: 'failed',
              error: {
                code: 'CREATION_FAILED',
                message: 'No se pudo crear el modelo',
                timestamp: new Date(),
                retryable: true
              },
              updatedAt: new Date()
            })
          }
          return updated
        })

        setCurrentProgress(null)
        return null
      } catch (error) {
        // Manejar error
        const modelError = error instanceof Error ? {
          code: 'VALIDATION_ERROR',
          message: error.message,
          timestamp: new Date(),
          retryable: false
        } : {
          code: 'UNKNOWN_ERROR',
          message: 'Error desconocido',
          timestamp: new Date(),
          retryable: false
        }

        setModelStates(prev => {
          const updated = new Map(prev)
          const current = updated.get(modelId)
          if (current) {
            updated.set(modelId, {
              ...current,
              status: 'failed',
              error: modelError,
              updatedAt: new Date()
            })
          }
          return updated
        })

        setCurrentProgress(null)
        return null
      }
    },
    [system]
  )

  // Validar descripción con resultado tipado
  const validateDescription = useCallback(
    (description: string): ValidationResult => {
      system.validateDescription(description)
      
      // Por ahora retornamos un resultado básico
      // En el futuro, esto debería usar el validador real
      return {
        valid: description.trim().length > 0,
        errors: description.trim().length === 0 
          ? [{ field: 'description', message: 'La descripción no puede estar vacía', code: 'EMPTY_DESCRIPTION', severity: 'error' }]
          : [],
        warnings: [],
        score: description.trim().length > 10 ? 100 : 50
      }
    },
    [system]
  )

  // Cancelar modelo
  const cancelModel = useCallback(
    async (modelId: string): Promise<boolean> => {
      const state = modelStates.get(modelId)
      if (!state || state.status === 'completed' || state.status === 'failed') {
        return false
      }

      setModelStates(prev => {
        const updated = new Map(prev)
        const current = updated.get(modelId)
        if (current) {
          updated.set(modelId, {
            ...current,
            status: 'cancelled',
            updatedAt: new Date()
          })
        }
        return updated
      })

      if (currentProgress?.modelId === modelId) {
        setCurrentProgress(null)
      }

      return true
    },
    [modelStates, currentProgress]
  )

  // Obtener estado de modelo
  const getModelState = useCallback(
    (modelId: string): ModelState | null => {
      return modelStates.get(modelId) || null
    },
    [modelStates]
  )

  // Obtener todos los estados
  const getAllModelStates = useCallback(
    (): readonly ModelState[] => {
      return Array.from(modelStates.values())
    },
    [modelStates]
  )

  // Limpiar todo
  const clearAll = useCallback(() => {
    system.clearAll()
    setModelStates(new Map())
    setCurrentProgress(null)
  }, [system])

  return {
    isCreating: system.isCreating,
    isValidationPending: system.isValidationPending,
    validationResult: system.validationResult as ValidationResult | null,
    activeModels: system.activeModels,
    currentProgress,
    createModel,
    validateDescription,
    cancelModel,
    getModelState,
    getAllModelStates,
    clearAll
  }
}

