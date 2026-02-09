/**
 * Test Helpers
 */

import { ProactiveBuildResult } from '@/components/ProactiveModelBuilder'

export const createMockModel = (overrides: Partial<ProactiveBuildResult> = {}): ProactiveBuildResult => {
  return {
    modelId: `model-${Date.now()}-${Math.random()}`,
    modelName: 'test-model',
    description: 'test description',
    status: 'completed',
    duration: 5000,
    startTime: Date.now() - 5000,
    endTime: Date.now(),
    ...overrides,
  }
}

export const createMockModels = (count: number, overrides: Partial<ProactiveBuildResult>[] = []): ProactiveBuildResult[] => {
  return Array(count).fill(null).map((_, i) => {
    const modelOverrides = overrides[i] || {}
    return createMockModel({
      modelId: `model-${i}`,
      modelName: `test-model-${i}`,
      description: `test description ${i}`,
      ...modelOverrides,
    })
  })
}

export const waitFor = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export const mockFetch = (response: any, ok: boolean = true) => {
  global.fetch = jest.fn().mockResolvedValue({
    ok,
    status: ok ? 200 : 500,
    json: async () => response,
    text: async () => JSON.stringify(response),
  })
}

export const mockLocalStorage = () => {
  const store: Record<string, string> = {}
  
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key]
    }),
    clear: jest.fn(() => {
      Object.keys(store).forEach(key => delete store[key])
    }),
  }
}

export const createMockEvent = (type: string, data: any = {}) => {
  return new CustomEvent(type, { detail: data })
}










