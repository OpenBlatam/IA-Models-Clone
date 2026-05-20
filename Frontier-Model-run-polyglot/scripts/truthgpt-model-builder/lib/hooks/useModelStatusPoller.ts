/**
 * Hook para polling de estado de modelos
 * =======================================
 * 
 * Maneja el polling del estado de modelos con reintentos,
 * manejo de errores y limpieza automática.
 */

import { useCallback, useRef } from 'react'
import { toast } from 'react-hot-toast'
import { TruthGPTAPIClient } from '../truthgpt-api-client'

export interface ModelStatus {
  status: 'creating' | 'completed' | 'failed'
  progress?: number
  currentStep?: string
  githubUrl?: string | null
  error?: string
}

export interface PollingOptions {
  modelId: string
  onStatusUpdate?: (status: ModelStatus) => void
  onComplete?: (status: ModelStatus) => void
  onError?: (error: Error) => void
  maxAttempts?: number
  pollInterval?: number
  immediate?: boolean
}

export interface UseModelStatusPollerResult {
  startPolling: (options: PollingOptions) => () => void // Retorna función para detener
  stopPolling: (modelId: string) => void
  isPolling: (modelId: string) => boolean
}

/**
 * Obtiene el estado del modelo usando TruthGPT API
 */
async function getStatusFromTruthGPTAPI(
  client: TruthGPTAPIClient,
  modelId: string
): Promise<ModelStatus> {
  try {
    const status = await client.getModelStatus(modelId)
    return {
      status: status.status === 'completed' ? 'completed' : 'creating',
      progress: status.progress ?? (status.status === 'completed' ? 100 : 50),
      currentStep: status.currentStep ?? 'Checking status...',
      githubUrl: status.githubUrl,
      error: status.error,
    }
  } catch (error) {
    // Si el modelo no existe, retornar estado de creación
    return {
      status: 'creating',
      progress: 0,
      currentStep: 'Initializing...',
    }
  }
}

/**
 * Obtiene el estado del modelo usando la API legacy
 */
async function getStatusFromLegacyAPI(modelId: string): Promise<ModelStatus> {
  const response = await fetch(`/api/model-status/${modelId}`)
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  
  const data = await response.json()
  
  return {
    status: data.status || 'creating',
    progress: data.progress ?? 0,
    currentStep: data.currentStep || 'Checking status...',
    githubUrl: data.githubUrl || null,
    error: data.error,
  }
}

/**
 * Hook para polling de estado de modelos
 */
export function useModelStatusPoller(
  apiClient: TruthGPTAPIClient | null,
  apiConnected: boolean
): UseModelStatusPollerResult {
  const pollingRefs = useRef<Map<string, { interval: NodeJS.Timeout; attempts: number }>>(new Map())
  
  const startPolling = useCallback(
    (options: PollingOptions): (() => void) => {
      const {
        modelId,
        onStatusUpdate,
        onComplete,
        onError,
        maxAttempts = 60, // 5 minutos máximo (60 intentos * 5 segundos)
        pollInterval = 5000, // 5 segundos
        immediate = false,
      } = options
      
      // Validar modelId
      if (!modelId || typeof modelId !== 'string' || modelId.trim().length === 0) {
        console.error('Invalid modelId for polling')
        toast.error('Error: ID de modelo inválido')
        return () => {}
      }
      
      // Detener polling anterior si existe
      if (pollingRefs.current.has(modelId)) {
        const existing = pollingRefs.current.get(modelId)!
        clearInterval(existing.interval)
        pollingRefs.current.delete(modelId)
      }
      
      let attempts = 0
      
      const checkStatus = async () => {
        try {
          attempts++
          
          // Verificar límite de intentos
          if (attempts > maxAttempts) {
            const error = new Error('Tiempo máximo de polling excedido')
            onError?.(error)
            stopPolling(modelId)
            toast.error('El polling del modelo excedió el tiempo máximo', { duration: 5000 })
            return
          }
          
          let status: ModelStatus
          
          // Intentar usar TruthGPT API si está disponible
          if (apiConnected && apiClient) {
            try {
              status = await getStatusFromTruthGPTAPI(apiClient, modelId)
            } catch (apiError) {
              console.warn('TruthGPT API status error, falling back to legacy API:', apiError)
              status = await getStatusFromLegacyAPI(modelId)
            }
          } else {
            status = await getStatusFromLegacyAPI(modelId)
          }
          
          // Notificar actualización de estado
          onStatusUpdate?.(status)
          
          // Si el modelo está completo o falló, detener polling
          if (status.status === 'completed' || status.status === 'failed') {
            stopPolling(modelId)
            onComplete?.(status)
            
            if (status.status === 'completed') {
              toast.success('Modelo completado exitosamente', { icon: '✅' })
            } else if (status.status === 'failed') {
              toast.error(status.error || 'El modelo falló', { duration: 5000 })
            }
          }
        } catch (error) {
          console.error('Error checking model status:', error)
          const errorMessage = error instanceof Error ? error.message : 'Error desconocido'
          
          // Solo mostrar error después de varios intentos fallidos
          if (attempts > 3) {
            onError?.(error instanceof Error ? error : new Error(errorMessage))
            // No detener el polling, solo notificar el error
          }
        }
      }
      
      // Ejecutar inmediatamente si se solicita
      if (immediate) {
        checkStatus()
      }
      
      // Configurar intervalo de polling
      const interval = setInterval(checkStatus, pollInterval)
      
      // Guardar referencia
      pollingRefs.current.set(modelId, { interval, attempts: 0 })
      
      // Retornar función para detener
      return () => stopPolling(modelId)
    },
    [apiClient, apiConnected]
  )
  
  const stopPolling = useCallback((modelId: string) => {
    const polling = pollingRefs.current.get(modelId)
    if (polling) {
      clearInterval(polling.interval)
      pollingRefs.current.delete(modelId)
    }
  }, [])
  
  const isPolling = useCallback((modelId: string): boolean => {
    return pollingRefs.current.has(modelId)
  }, [])
  
  return {
    startPolling,
    stopPolling,
    isPolling,
  }
}

