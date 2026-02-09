import { useState, useCallback } from 'react'
import { api } from '@/lib/api'
import { getErrorMessage } from '@/lib/utils'

interface UseGeneratorControlReturn {
  start: () => Promise<boolean>
  stop: () => Promise<boolean>
  loading: boolean
  error: string | null
  clearError: () => void
}

export const useGeneratorControl = (): UseGeneratorControlReturn => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const start = useCallback(async (): Promise<boolean> => {
    setLoading(true)
    setError(null)

    try {
      await api.startGenerator()
      return true
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      setError(errorMessage)
      console.error('Error starting generator:', err)
      return false
    } finally {
      setLoading(false)
    }
  }, [])

  const stop = useCallback(async (): Promise<boolean> => {
    setLoading(true)
    setError(null)

    try {
      await api.stopGenerator()
      return true
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      setError(errorMessage)
      console.error('Error stopping generator:', err)
      return false
    } finally {
      setLoading(false)
    }
  }, [])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
    start,
    stop,
    loading,
    error,
    clearError,
  }
}

