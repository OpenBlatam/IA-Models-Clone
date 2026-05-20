/**
 * Hook useAsync
 * =============
 * 
 * Hook para manejar operaciones asíncronas
 */

import { useState, useEffect, useCallback, useRef } from 'react'

export interface UseAsyncOptions<T> {
  immediate?: boolean
  onSuccess?: (data: T) => void
  onError?: (error: Error) => void
}

export interface UseAsyncResult<T> {
  data: T | null
  error: Error | null
  loading: boolean
  execute: (...args: any[]) => Promise<T | undefined>
  reset: () => void
}

/**
 * Hook para manejar operaciones asíncronas
 */
export function useAsync<T>(
  asyncFunction: (...args: any[]) => Promise<T>,
  options: UseAsyncOptions<T> = {}
): UseAsyncResult<T> {
  const { immediate = false, onSuccess, onError } = options
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(immediate)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const execute = useCallback(
    async (...args: any[]): Promise<T | undefined> => {
      setLoading(true)
      setError(null)

      try {
        const result = await asyncFunction(...args)
        
        if (mountedRef.current) {
          setData(result)
          setLoading(false)
          onSuccess?.(result)
          return result
        }
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err))
        
        if (mountedRef.current) {
          setError(error)
          setLoading(false)
          onError?.(error)
        }
      }
    },
    [asyncFunction, onSuccess, onError]
  )

  const reset = useCallback(() => {
    setData(null)
    setError(null)
    setLoading(false)
  }, [])

  useEffect(() => {
    if (immediate) {
      execute()
    }
  }, []) // Solo ejecutar una vez

  return {
    data,
    error,
    loading,
    execute,
    reset
  }
}







