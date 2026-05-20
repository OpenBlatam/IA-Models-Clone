/**
 * Hook para Manejo de Errores
 * ===========================
 * 
 * Hook centralizado para manejar errores de manera consistente
 */

import { useCallback, useState } from 'react'
import { ModelError } from '../types/modelTypes'

export interface ErrorHandlerOptions {
  onError?: (error: Error | ModelError) => void
  showNotifications?: boolean
  logErrors?: boolean
}

export interface UseErrorHandlerResult {
  error: Error | ModelError | null
  handleError: (error: unknown) => void
  clearError: () => void
  isError: boolean
}

/**
 * Hook para manejar errores de manera centralizada
 */
export function useErrorHandler(
  options: ErrorHandlerOptions = {}
): UseErrorHandlerResult {
  const {
    onError,
    showNotifications = true,
    logErrors = true
  } = options

  const [error, setError] = useState<Error | ModelError | null>(null)

  const handleError = useCallback((err: unknown) => {
    let processedError: Error | ModelError

    if (err instanceof Error) {
      processedError = err
    } else if (err && typeof err === 'object' && 'code' in err && 'message' in err) {
      processedError = err as ModelError
    } else {
      processedError = new Error(String(err || 'Error desconocido'))
    }

    setError(processedError)

    // Log error
    if (logErrors) {
      console.error('Error handled:', processedError)
    }

    // Mostrar notificación
    if (showNotifications && typeof window !== 'undefined') {
      // Usar toast si está disponible
      if (typeof window !== 'undefined' && 'toast' in window) {
        // @ts-ignore
        window.toast?.error(processedError.message)
      }
    }

    // Callback personalizado
    onError?.(processedError)
  }, [onError, showNotifications, logErrors])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
    error,
    handleError,
    clearError,
    isError: error !== null
  }
}







