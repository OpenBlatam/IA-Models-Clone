/**
 * Hook para usar el API Client Mejorado
 * =====================================
 * 
 * Proporciona una instancia del API client con configuración y estado
 */

import { useMemo, useCallback, useState, useEffect } from 'react'
import { TruthGPTAPIClientEnhanced, APIClientOptions } from '../truthgpt-api-client-enhanced'
import { ModelSpec } from '../types/modelTypes'

export interface UseAPIClientOptions extends APIClientOptions {
  autoConnect?: boolean
  onError?: (error: Error) => void
  onSuccess?: (message: string) => void
}

export interface UseAPIClientResult {
  client: TruthGPTAPIClientEnhanced
  isConnected: boolean
  isConnecting: boolean
  error: Error | null
  connect: () => Promise<boolean>
  disconnect: () => void
  clearCache: () => void
  getCacheStats: () => { size: number; maxSize: number; hitRate: number }
}

/**
 * Hook para usar el API client mejorado
 */
export function useAPIClient(
  options: UseAPIClientOptions = {}
): UseAPIClientResult {
  const {
    autoConnect = true,
    onError,
    onSuccess,
    ...clientOptions
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  // Crear instancia del client
  const client = useMemo(() => {
    const instance = new TruthGPTAPIClientEnhanced(clientOptions)

    // Agregar interceptor de error global
    instance.addErrorInterceptor((err) => {
      setError(err)
      onError?.(err)
      return err
    })

    // Agregar interceptor de response para logging
    instance.addResponseInterceptor((response) => {
      if (response.status >= 200 && response.status < 300) {
        onSuccess?.(`Request successful: ${response.status}`)
      }
      return response
    })

    return instance
  }, [JSON.stringify(clientOptions), onError, onSuccess])

  // Conectar al API
  const connect = useCallback(async (): Promise<boolean> => {
    setIsConnecting(true)
    setError(null)

    try {
      const health = await client.healthCheck()
      setIsConnected(health.status === 'ok' || health.status === 'healthy')
      setIsConnecting(false)
      return isConnected
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      setError(error)
      setIsConnected(false)
      setIsConnecting(false)
      onError?.(error)
      return false
    }
  }, [client, isConnected, onError])

  // Desconectar
  const disconnect = useCallback(() => {
    setIsConnected(false)
    setError(null)
    client.clearCache()
  }, [client])

  // Limpiar cache
  const clearCache = useCallback(() => {
    client.clearCache()
  }, [client])

  // Obtener estadísticas de cache
  const getCacheStats = useCallback(() => {
    return client.getCacheStats()
  }, [client])

  // Auto-conectar si está habilitado
  useEffect(() => {
    if (autoConnect && !isConnected && !isConnecting) {
      connect()
    }
  }, [autoConnect, isConnected, isConnecting, connect])

  return {
    client,
    isConnected,
    isConnecting,
    error,
    connect,
    disconnect,
    clearCache,
    getCacheStats
  }
}







