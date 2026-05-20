/**
 * Hook para estrategias de reintento inteligentes
 * ===============================================
 */

import { useState, useCallback } from 'react'

export type RetryStrategy = 'exponential' | 'linear' | 'fixed' | 'custom'

export interface RetryOptions {
  maxRetries?: number
  initialDelay?: number
  maxDelay?: number
  strategy?: RetryStrategy
  customDelays?: number[]
  onRetry?: (attempt: number, error: Error) => void
  shouldRetry?: (error: Error, attempt: number) => boolean
}

export interface RetryResult<T> {
  result: T | null
  attempts: number
  errors: Error[]
}

/**
 * Calcula el delay según la estrategia
 */
function calculateDelay(
  attempt: number,
  initialDelay: number,
  maxDelay: number,
  strategy: RetryStrategy,
  customDelays?: number[]
): number {
  if (strategy === 'custom' && customDelays) {
    return customDelays[attempt - 1] || customDelays[customDelays.length - 1]
  }

  let delay: number

  switch (strategy) {
    case 'exponential':
      delay = initialDelay * Math.pow(2, attempt - 1)
      break
    case 'linear':
      delay = initialDelay * attempt
      break
    case 'fixed':
      delay = initialDelay
      break
    default:
      delay = initialDelay
  }

  return Math.min(delay, maxDelay)
}

/**
 * Hook para ejecutar operaciones con reintentos inteligentes
 */
export function useModelRetry() {
  const [retryState, setRetryState] = useState<{
    isRetrying: boolean
    currentAttempt: number
    lastError: Error | null
  }>({
    isRetrying: false,
    currentAttempt: 0,
    lastError: null
  })

  const retry = useCallback(
    async <T>(
      operation: () => Promise<T>,
      options: RetryOptions = {}
    ): Promise<RetryResult<T>> => {
      const {
        maxRetries = 3,
        initialDelay = 1000,
        maxDelay = 30000,
        strategy = 'exponential',
        customDelays,
        onRetry,
        shouldRetry
      } = options

      const errors: Error[] = []
      let lastResult: T | null = null

      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        setRetryState({
          isRetrying: attempt > 1,
          currentAttempt: attempt,
          lastError: null
        })

        try {
          lastResult = await operation()
          
          setRetryState({
            isRetrying: false,
            currentAttempt: 0,
            lastError: null
          })

          return {
            result: lastResult,
            attempts: attempt,
            errors
          }
        } catch (error) {
          const err = error instanceof Error ? error : new Error(String(error))
          errors.push(err)

          setRetryState({
            isRetrying: attempt < maxRetries,
            currentAttempt: attempt,
            lastError: err
          })

          // Verificar si debe reintentar
          if (shouldRetry && !shouldRetry(err, attempt)) {
            break
          }

          // Si es el último intento, no esperar
          if (attempt >= maxRetries) {
            break
          }

          // Notificar reintento
          onRetry?.(attempt, err)

          // Esperar antes del siguiente intento
          const delay = calculateDelay(attempt, initialDelay, maxDelay, strategy, customDelays)
          await new Promise(resolve => setTimeout(resolve, delay))
        }
      }

      setRetryState({
        isRetrying: false,
        currentAttempt: 0,
        lastError: errors[errors.length - 1] || null
      })

      return {
        result: lastResult,
        attempts: errors.length,
        errors
      }
    },
    []
  )

  return {
    retry,
    isRetrying: retryState.isRetrying,
    currentAttempt: retryState.currentAttempt,
    lastError: retryState.lastError
  }
}

