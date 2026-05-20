/**
 * Hook para cola de creación de modelos
 * ======================================
 */

import { useState, useCallback, useRef } from 'react'

export interface QueuedModel {
  id: string
  description: string
  spec?: any
  priority: number
  createdAt: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  error?: Error
}

export interface UseModelQueueOptions {
  maxConcurrent?: number
  autoProcess?: boolean
}

export interface UseModelQueueResult {
  enqueue: (description: string, spec?: any, priority?: number) => string
  dequeue: (id: string) => QueuedModel | null
  clear: () => void
  queue: QueuedModel[]
  processing: QueuedModel[]
  completed: QueuedModel[]
  failed: QueuedModel[]
  stats: {
    total: number
    pending: number
    processing: number
    completed: number
    failed: number
  }
}

/**
 * Hook para gestionar una cola de creación de modelos
 */
export function useModelQueue(
  onCreateModel: (description: string, spec?: any) => Promise<string | null>,
  options: UseModelQueueOptions = {}
): UseModelQueueResult {
  const {
    maxConcurrent = 1,
    autoProcess = true
  } = options

  const [queue, setQueue] = useState<QueuedModel[]>([])
  const processingRef = useRef<Set<string>>(new Set())

  const enqueue = useCallback((
    description: string,
    spec?: any,
    priority: number = 0
  ): string => {
    const id = `queue-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const model: QueuedModel = {
      id,
      description,
      spec,
      priority,
      createdAt: Date.now(),
      status: 'pending'
    }

    setQueue(prev => {
      const next = [...prev, model]
      // Ordenar por prioridad (mayor primero) y luego por fecha
      next.sort((a, b) => {
        if (a.priority !== b.priority) {
          return b.priority - a.priority
        }
        return a.createdAt - b.createdAt
      })
      return next
    })

    // Procesar automáticamente si está habilitado
    if (autoProcess) {
      processQueue()
    }

    return id
  }, [autoProcess])

  const dequeue = useCallback((id: string): QueuedModel | null => {
    let found: QueuedModel | null = null

    setQueue(prev => {
      const index = prev.findIndex(m => m.id === id)
      if (index === -1) return prev

      found = prev[index]
      return prev.filter(m => m.id !== id)
    })

    return found
  }, [])

  const processQueue = useCallback(async () => {
    // Verificar si hay espacio para procesar más
    if (processingRef.current.size >= maxConcurrent) {
      return
    }

    // Obtener el siguiente modelo pendiente
    const pending = queue
      .filter(m => m.status === 'pending')
      .sort((a, b) => {
        if (a.priority !== b.priority) {
          return b.priority - a.priority
        }
        return a.createdAt - b.createdAt
      })[0]

    if (!pending) {
      return
    }

    // Marcar como procesando
    processingRef.current.add(pending.id)
    setQueue(prev =>
      prev.map(m =>
        m.id === pending.id ? { ...m, status: 'processing' } : m
      )
    )

    try {
      const modelId = await onCreateModel(pending.description, pending.spec)

      setQueue(prev =>
        prev.map(m =>
          m.id === pending.id
            ? { ...m, status: 'completed' }
            : m
        )
      )
    } catch (error) {
      setQueue(prev =>
        prev.map(m =>
          m.id === pending.id
            ? {
                ...m,
                status: 'failed',
                error: error instanceof Error ? error : new Error(String(error))
              }
            : m
        )
      )
    } finally {
      processingRef.current.delete(pending.id)
      
      // Procesar siguiente si hay espacio
      if (autoProcess) {
        setTimeout(() => processQueue(), 100)
      }
    }
  }, [queue, maxConcurrent, onCreateModel, autoProcess])

  const clear = useCallback(() => {
    setQueue([])
    processingRef.current.clear()
  }, [])

  const stats = {
    total: queue.length,
    pending: queue.filter(m => m.status === 'pending').length,
    processing: queue.filter(m => m.status === 'processing').length,
    completed: queue.filter(m => m.status === 'completed').length,
    failed: queue.filter(m => m.status === 'failed').length
  }

  return {
    enqueue,
    dequeue,
    clear,
    queue,
    processing: queue.filter(m => m.status === 'processing'),
    completed: queue.filter(m => m.status === 'completed'),
    failed: queue.filter(m => m.status === 'failed'),
    stats
  }
}

