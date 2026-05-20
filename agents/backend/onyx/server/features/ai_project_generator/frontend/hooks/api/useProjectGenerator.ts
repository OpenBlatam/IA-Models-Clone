import { useState, useCallback } from 'react'
import { api } from '@/lib/api'
import { getErrorMessage } from '@/lib/utils'
import type { ProjectRequest } from '@/types'

interface UseProjectGeneratorReturn {
  generate: (request: ProjectRequest) => Promise<string | null>
  loading: boolean
  error: string | null
  clearError: () => void
}

export const useProjectGenerator = (): UseProjectGeneratorReturn => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const generate = useCallback(async (request: ProjectRequest): Promise<string | null> => {
    setLoading(true)
    setError(null)

    try {
      const response = await api.generate(request)
      return response.project_id
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      setError(errorMessage || 'Failed to generate project')
      console.error('Error generating project:', err)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
    generate,
    loading,
    error,
    clearError,
  }
}

