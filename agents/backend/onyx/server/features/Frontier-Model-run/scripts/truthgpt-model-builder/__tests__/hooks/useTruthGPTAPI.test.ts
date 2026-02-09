/**
 * Unit Tests - useTruthGPTAPI Hook
 */

import { renderHook, waitFor } from '@testing-library/react'
import { useTruthGPTAPI } from '@/lib/hooks'

// Mock fetch
global.fetch = jest.fn()

describe('useTruthGPTAPI', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  describe('API Calls', () => {
    it('should create model via API', async () => {
      const mockFetch = global.fetch as jest.Mock
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          modelId: 'test-model-123',
          status: 'creating',
        }),
      })

      const { result } = renderHook(() => useTruthGPTAPI())

      const response = await result.current.createModel({
        description: 'test model',
        modelName: 'test-model',
      })

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/create-model'),
        expect.objectContaining({
          method: 'POST',
        })
      )
      expect(response.modelId).toBe('test-model-123')
    })

    it('should get model status', async () => {
      const mockFetch = global.fetch as jest.Mock
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          modelId: 'test-model-123',
          status: 'completed',
        }),
      })

      const { result } = renderHook(() => useTruthGPTAPI())

      const status = await result.current.getModelStatus('test-model-123')

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/model-status'),
        expect.objectContaining({
          method: 'GET',
        })
      )
      expect(status.status).toBe('completed')
    })

    it('should handle API errors', async () => {
      const mockFetch = global.fetch as jest.Mock
      mockFetch.mockRejectedValueOnce(new Error('Network error'))

      const { result } = renderHook(() => useTruthGPTAPI())

      await expect(
        result.current.createModel({
          description: 'test',
          modelName: 'test',
        })
      ).rejects.toThrow()
    })

    it('should handle non-OK responses', async () => {
      const mockFetch = global.fetch as jest.Mock
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ error: 'Bad Request' }),
      })

      const { result } = renderHook(() => useTruthGPTAPI())

      await expect(
        result.current.createModel({
          description: 'test',
          modelName: 'test',
        })
      ).rejects.toThrow()
    })
  })

  describe('Rate Limiting', () => {
    it('should track rate limit', async () => {
      const mockFetch = global.fetch as jest.Mock
      mockFetch.mockResolvedValue({
        ok: true,
        headers: new Headers({
          'X-RateLimit-Remaining': '10',
          'X-RateLimit-Reset': String(Date.now() + 60000),
        }),
        json: async () => ({ modelId: 'test' }),
      })

      const { result } = renderHook(() => useTruthGPTAPI())

      await result.current.createModel({
        description: 'test',
        modelName: 'test',
      })

      // Rate limit should be tracked
      expect(result.current.rateLimit).toBeDefined()
    })
  })

  describe('Retry Logic', () => {
    it('should retry on failure', async () => {
      const mockFetch = global.fetch as jest.Mock
      mockFetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ modelId: 'test' }),
        })

      const { result } = renderHook(() => useTruthGPTAPI())

      const response = await result.current.createModel({
        description: 'test',
        modelName: 'test',
      })

      expect(mockFetch).toHaveBeenCalledTimes(2)
      expect(response.modelId).toBe('test')
    })
  })
})










