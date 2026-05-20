/**
 * Hook para Observar Modelos
 * ==========================
 * 
 * Observa cambios en el estado de modelos y ejecuta callbacks
 */

import { useEffect, useRef, useCallback } from 'react'
import { ModelState, ModelStatus } from '../types/modelTypes'

export interface ModelWatcherOptions {
  onStatusChange?: (modelId: string, oldStatus: ModelStatus, newStatus: ModelStatus) => void
  onProgressChange?: (modelId: string, progress: number) => void
  onError?: (modelId: string, error: Error) => void
  onComplete?: (modelId: string, state: ModelState) => void
  pollInterval?: number
}

export interface UseModelWatcherResult {
  watch: (modelId: string, state: ModelState) => void
  unwatch: (modelId: string) => void
  getWatched: () => Set<string>
  clear: () => void
}

/**
 * Hook para observar cambios en modelos
 */
export function useModelWatcher(
  options: ModelWatcherOptions = {}
): UseModelWatcherResult {
  const {
    onStatusChange,
    onProgressChange,
    onError,
    onComplete,
    pollInterval = 1000
  } = options

  const watchedModels = useRef<Map<string, ModelState>>(new Map())
  const callbacksRef = useRef(options)

  // Actualizar callbacks
  useEffect(() => {
    callbacksRef.current = options
  }, [options])

  // Observar cambios
  const watch = useCallback((modelId: string, state: ModelState) => {
    const previous = watchedModels.current.get(modelId)

    // Estado anterior no existe, agregar
    if (!previous) {
      watchedModels.current.set(modelId, state)
      return
    }

    // Estado cambió
    if (previous.status !== state.status) {
      callbacksRef.current.onStatusChange?.(modelId, previous.status, state.status)
    }

    // Progreso cambió
    if (previous.progress !== state.progress) {
      callbacksRef.current.onProgressChange?.(modelId, state.progress)
    }

    // Error ocurrió
    if (!previous.error && state.error) {
      callbacksRef.current.onError?.(modelId, new Error(state.error.message))
    }

    // Completado
    if (state.status === 'completed' && previous.status !== 'completed') {
      callbacksRef.current.onComplete?.(modelId, state)
    }

    // Actualizar estado
    watchedModels.current.set(modelId, state)
  }, [])

  // Dejar de observar
  const unwatch = useCallback((modelId: string) => {
    watchedModels.current.delete(modelId)
  }, [])

  // Obtener modelos observados
  const getWatched = useCallback(() => {
    return new Set(watchedModels.current.keys())
  }, [])

  // Limpiar todo
  const clear = useCallback(() => {
    watchedModels.current.clear()
  }, [])

  return {
    watch,
    unwatch,
    getWatched,
    clear
  }
}







