/**
 * React Hook for TruthGPT API
 * ============================
 * 
 * Hook para usar la API REST de TruthGPT en componentes React
 */

import { useState, useCallback } from 'react'
import { getTruthGPTAPIClient, TruthGPTAPIClient } from '../truthgpt-api-client'

export interface UseTruthGPTAPIResult {
  client: TruthGPTAPIClient
  isConnected: boolean
  checkConnection: () => Promise<boolean>
  createModelFromDescription: (description: string, modelName?: string) => Promise<{ modelId: string; name: string }>
  isLoading: boolean
  error: string | null
}

export function useTruthGPTAPI(baseUrl?: string): UseTruthGPTAPIResult {
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const client = getTruthGPTAPIClient(baseUrl)

  const checkConnection = useCallback(async (): Promise<boolean> => {
    setIsLoading(true)
    setError(null)
    try {
      await client.healthCheck()
      setIsConnected(true)
      return true
    } catch (err) {
      setIsConnected(false)
      setError(err instanceof Error ? err.message : 'Connection failed')
      return false
    } finally {
      setIsLoading(false)
    }
  }, [client])

  const createModelFromDescription = useCallback(
    async (description: string, modelName?: string): Promise<{ modelId: string; name: string }> => {
      setIsLoading(true)
      setError(null)
      try {
        const result = await client.createModelFromDescription(description, modelName)
        return result
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to create model'
        setError(errorMessage)
        throw err
      } finally {
        setIsLoading(false)
      }
    },
    [client]
  )

  return {
    client,
    isConnected,
    checkConnection,
    createModelFromDescription,
    isLoading,
    error,
  }
}

