/**
 * Hook para gestión avanzada de historial de modelos
 * ====================================================
 */

import { useState, useCallback, useEffect, useMemo } from 'react'

export interface ModelHistoryItem {
  id: string
  name: string
  description: string
  spec?: any
  status: 'creating' | 'completed' | 'failed'
  createdAt: number
  completedAt?: number
  duration?: number
  githubUrl?: string
  error?: string
  tags?: string[]
  notes?: string
}

export interface UseModelHistoryOptions {
  maxItems?: number
  enablePersistence?: boolean
  storageKey?: string
}

export interface UseModelHistoryResult {
  history: ModelHistoryItem[]
  addModel: (model: Omit<ModelHistoryItem, 'id' | 'createdAt'>) => string
  updateModel: (id: string, updates: Partial<ModelHistoryItem>) => void
  removeModel: (id: string) => void
  getModel: (id: string) => ModelHistoryItem | undefined
  searchModels: (query: string) => ModelHistoryItem[]
  filterByStatus: (status: ModelHistoryItem['status']) => ModelHistoryItem[]
  filterByTags: (tags: string[]) => ModelHistoryItem[]
  getStats: () => {
    total: number
    completed: number
    failed: number
    creating: number
    averageDuration: number
  }
  clear: () => void
  exportHistory: () => string
}

/**
 * Hook para gestión de historial de modelos con búsqueda y filtrado
 */
export function useModelHistory(
  options: UseModelHistoryOptions = {}
): UseModelHistoryResult {
  const {
    maxItems = 100,
    enablePersistence = true,
    storageKey = 'truthgpt-model-history'
  } = options

  const [history, setHistory] = useState<ModelHistoryItem[]>(() => {
    if (enablePersistence && typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem(storageKey)
        if (stored) {
          return JSON.parse(stored)
        }
      } catch (error) {
        console.error('Error loading model history:', error)
      }
    }
    return []
  })

  // Persistir en localStorage
  useEffect(() => {
    if (enablePersistence && typeof window !== 'undefined') {
      try {
        localStorage.setItem(storageKey, JSON.stringify(history))
      } catch (error) {
        console.error('Error saving model history:', error)
      }
    }
  }, [history, enablePersistence, storageKey])

  const addModel = useCallback((
    model: Omit<ModelHistoryItem, 'id' | 'createdAt'>
  ): string => {
    const id = `model-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const newModel: ModelHistoryItem = {
      ...model,
      id,
      createdAt: Date.now()
    }

    setHistory(prev => {
      const updated = [newModel, ...prev]
      // Limitar tamaño
      return updated.slice(0, maxItems)
    })

    return id
  }, [maxItems])

  const updateModel = useCallback((
    id: string,
    updates: Partial<ModelHistoryItem>
  ) => {
    setHistory(prev => prev.map(model => {
      if (model.id === id) {
        const updated = { ...model, ...updates }
        
        // Calcular duración si se completó
        if (updates.status === 'completed' && !updated.duration && updated.completedAt) {
          updated.duration = updated.completedAt - model.createdAt
        }

        return updated
      }
      return model
    }))
  }, [])

  const removeModel = useCallback((id: string) => {
    setHistory(prev => prev.filter(model => model.id !== id))
  }, [])

  const getModel = useCallback((id: string) => {
    return history.find(model => model.id === id)
  }, [history])

  const searchModels = useCallback((query: string) => {
    const lowerQuery = query.toLowerCase()
    return history.filter(model =>
      model.name.toLowerCase().includes(lowerQuery) ||
      model.description.toLowerCase().includes(lowerQuery) ||
      model.tags?.some(tag => tag.toLowerCase().includes(lowerQuery))
    )
  }, [history])

  const filterByStatus = useCallback((status: ModelHistoryItem['status']) => {
    return history.filter(model => model.status === status)
  }, [history])

  const filterByTags = useCallback((tags: string[]) => {
    return history.filter(model =>
      model.tags?.some(tag => tags.includes(tag))
    )
  }, [history])

  const getStats = useCallback(() => {
    const completed = history.filter(m => m.status === 'completed')
    const durations = completed
      .map(m => m.duration || 0)
      .filter(d => d > 0)

    const averageDuration = durations.length > 0
      ? durations.reduce((a, b) => a + b, 0) / durations.length
      : 0

    return {
      total: history.length,
      completed: completed.length,
      failed: history.filter(m => m.status === 'failed').length,
      creating: history.filter(m => m.status === 'creating').length,
      averageDuration
    }
  }, [history])

  const clear = useCallback(() => {
    setHistory([])
    if (enablePersistence && typeof window !== 'undefined') {
      localStorage.removeItem(storageKey)
    }
  }, [enablePersistence, storageKey])

  const exportHistory = useCallback(() => {
    return JSON.stringify({
      history,
      stats: getStats(),
      exportedAt: new Date().toISOString()
    }, null, 2)
  }, [history, getStats])

  return {
    history,
    addModel,
    updateModel,
    removeModel,
    getModel,
    searchModels,
    filterByStatus,
    filterByTags,
    getStats,
    clear,
    exportHistory
  }
}

