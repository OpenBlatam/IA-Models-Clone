/**
 * Hook personalizado para creación y gestión de modelos
 * ======================================================
 * 
 * Consolida la lógica de creación de modelos, manejo de errores,
 * reintentos y polling de estado.
 */

import { useState, useCallback, useRef } from 'react'
import { toast } from 'react-hot-toast'
import { TruthGPTAPIClient } from '../truthgpt-api-client'
import { classifyModelError, getFriendlyErrorMessage, isRecoverableError } from '../modelErrorHandler'

export interface ModelCreationOptions {
  description: string
  modelName?: string
  spec?: any
  onSuccess?: (modelId: string, modelName: string) => void
  onError?: (error: Error) => void
  maxRetries?: number
  retryDelay?: number
}

export interface ModelCreationResult {
  modelId: string
  modelName: string
  description: string
  status: 'creating' | 'completed' | 'failed'
  githubUrl?: string | null
}

export interface UseModelCreatorResult {
  createModel: (options: ModelCreationOptions) => Promise<ModelCreationResult | null>
  isCreating: boolean
  error: string | null
  cancel: () => void
}

/**
 * Valida los parámetros de entrada para la creación de modelos
 */
function validateModelInput(description: string, spec: any): { valid: boolean; error?: string } {
  if (!description || typeof description !== 'string') {
    return { valid: false, error: 'Mensaje de usuario inválido' }
  }
  
  const trimmed = description.trim()
  if (trimmed.length === 0) {
    return { valid: false, error: 'El mensaje del usuario está vacío' }
  }
  
  if (trimmed.length < 10) {
    return { valid: false, error: 'La descripción debe tener al menos 10 caracteres' }
  }
  
  if (trimmed.length > 5000) {
    return { valid: false, error: 'La descripción es demasiado larga (máximo 5000 caracteres)' }
  }
  
  if (spec !== null && spec !== undefined && (typeof spec !== 'object' || Array.isArray(spec))) {
    console.warn('Invalid spec provided, using null')
  }
  
  return { valid: true }
}

/**
 * Intenta crear un modelo usando la API REST de TruthGPT
 */
async function createWithTruthGPTAPI(
  client: TruthGPTAPIClient,
  description: string,
  modelName: string
): Promise<ModelCreationResult> {
  const result = await client.createModelFromDescription(description, modelName)
  
  return {
    modelId: result.modelId,
    modelName: result.name,
    description,
    status: 'creating',
    githubUrl: null,
  }
}

/**
 * Intenta crear un modelo usando la API legacy
 */
async function createWithLegacyAPI(
  description: string,
  timeout: number = 60000
): Promise<ModelCreationResult> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)
  
  try {
    const response = await fetch('/api/create-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description: description.trim() }),
      signal: controller.signal,
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      let errorMessage = 'Error al crear el modelo'
      try {
        const errorData = await response.json()
        errorMessage = errorData.error || errorMessage
      } catch {
        errorMessage = `HTTP error! status: ${response.status}`
      }
      throw new Error(errorMessage)
    }
    
    const responseData = await response.json()
    
    if (!responseData || typeof responseData !== 'object') {
      throw new Error('Respuesta del servidor inválida')
    }
    
    if (!responseData.modelId || !responseData.modelName) {
      throw new Error('Respuesta del servidor incompleta: faltan modelId o modelName')
    }
    
    return {
      modelId: String(responseData.modelId),
      modelName: String(responseData.modelName),
      description,
      status: 'creating',
      githubUrl: responseData.githubUrl || null,
    }
  } catch (fetchError) {
    clearTimeout(timeoutId)
    if (fetchError instanceof Error && fetchError.name === 'AbortError') {
      throw new Error('Timeout: La solicitud tardó demasiado tiempo')
    }
    throw fetchError
  }
}

/**
 * Crea un modelo con reintentos automáticos
 */
async function createModelWithRetry(
  options: ModelCreationOptions,
  apiClient: TruthGPTAPIClient | null,
  apiConnected: boolean,
  maxRetries: number = 3,
  retryDelay: number = 1000
): Promise<ModelCreationResult> {
  const { description, modelName, spec } = options
  const trimmedDescription = description.trim()
  const finalModelName = modelName || spec?.modelName || `truthgpt-model-${Date.now()}`
  
  let lastError: Error | null = null
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Intentar primero con TruthGPT API si está disponible
      if (apiConnected && apiClient) {
        try {
          return await createWithTruthGPTAPI(apiClient, trimmedDescription, finalModelName)
        } catch (apiError) {
          console.warn(`TruthGPT API attempt ${attempt + 1} failed:`, apiError)
          lastError = apiError instanceof Error ? apiError : new Error(String(apiError))
          
          // Si no es el último intento, esperar antes de reintentar
          if (attempt < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, retryDelay * (attempt + 1)))
            continue
          }
          
          // Si falló TruthGPT API y es el último intento, usar legacy API como fallback
          console.warn('TruthGPT API failed, falling back to legacy API')
        }
      }
      
      // Usar legacy API
      return await createWithLegacyAPI(trimmedDescription)
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error))
      console.error(`Model creation attempt ${attempt + 1} failed:`, lastError)
      
      // Si no es el último intento, esperar antes de reintentar
      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, retryDelay * (attempt + 1)))
      }
    }
  }
  
  // Si todos los intentos fallaron, lanzar el último error
  throw lastError || new Error('Error desconocido al crear el modelo')
}

/**
 * Hook para crear modelos con manejo mejorado de errores y reintentos
 */
export function useModelCreator(
  apiClient: TruthGPTAPIClient | null,
  apiConnected: boolean
): UseModelCreatorResult {
  const [isCreating, setIsCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  
  const createModel = useCallback(
    async (options: ModelCreationOptions): Promise<ModelCreationResult | null> => {
      // Validar entrada
      const validation = validateModelInput(options.description, options.spec)
      if (!validation.valid) {
        const errorMsg = validation.error || 'Entrada inválida'
        toast.error(errorMsg)
        setError(errorMsg)
        options.onError?.(new Error(errorMsg))
        return null
      }
      
      setIsCreating(true)
      setError(null)
      
      // Crear AbortController para cancelación
      abortControllerRef.current = new AbortController()
      
      try {
        const maxRetries = options.maxRetries ?? 3
        const retryDelay = options.retryDelay ?? 1000
        
        const result = await createModelWithRetry(
          options,
          apiClient,
          apiConnected,
          maxRetries,
          retryDelay
        )
        
        // Validar resultado
        if (!result.modelId || !result.modelName) {
          throw new Error('Resultado inválido: faltan modelId o modelName')
        }
        
        toast.success('Modelo en creación', { icon: '🚀' })
        options.onSuccess?.(result.modelId, result.modelName)
        
        return result
      } catch (err) {
        const modelError = classifyModelError(err)
        const friendlyMessage = getFriendlyErrorMessage(modelError)
        
        setError(friendlyMessage)
        
        // Mostrar mensaje apropiado según el tipo de error
        const icon = modelError.type === 'RATE_LIMIT_ERROR' ? '⏱️' : 
                    modelError.type === 'TIMEOUT_ERROR' ? '⏳' :
                    modelError.type === 'NETWORK_ERROR' ? '🌐' : '❌'
        
        toast.error(friendlyMessage, { 
          icon,
          duration: modelError.retryAfter ? modelError.retryAfter * 1000 : 5000 
        })
        
        options.onError?.(modelError.originalError || new Error(friendlyMessage))
        return null
      } finally {
        setIsCreating(false)
        abortControllerRef.current = null
      }
    },
    [apiClient, apiConnected]
  )
  
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
      setIsCreating(false)
      setError('Operación cancelada')
    }
  }, [])
  
  return {
    createModel,
    isCreating,
    error,
    cancel,
  }
}

