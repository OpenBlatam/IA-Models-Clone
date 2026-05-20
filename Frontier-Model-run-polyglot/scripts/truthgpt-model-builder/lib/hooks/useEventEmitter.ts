/**
 * Hook useEventEmitter
 * ====================
 * 
 * Hook para crear y manejar emisores de eventos
 */

import { useCallback, useRef, useEffect } from 'react'

type EventHandler<T = any> = (data: T) => void

class EventEmitter<T extends Record<string, any> = Record<string, any>> {
  private handlers: Map<keyof T, Set<EventHandler>> = new Map()

  on<K extends keyof T>(event: K, handler: EventHandler<T[K]>): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set())
    }
    this.handlers.get(event)!.add(handler)

    // Retorna función para desuscribirse
    return () => {
      this.off(event, handler)
    }
  }

  off<K extends keyof T>(event: K, handler: EventHandler<T[K]>): void {
    const handlers = this.handlers.get(event)
    if (handlers) {
      handlers.delete(handler)
      if (handlers.size === 0) {
        this.handlers.delete(event)
      }
    }
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    const handlers = this.handlers.get(event)
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data)
        } catch (error) {
          console.error(`Error en handler de evento '${String(event)}':`, error)
        }
      })
    }
  }

  once<K extends keyof T>(event: K, handler: EventHandler<T[K]>): () => void {
    const wrappedHandler: EventHandler<T[K]> = (data) => {
      handler(data)
      this.off(event, wrappedHandler)
    }
    return this.on(event, wrappedHandler)
  }

  removeAllListeners<K extends keyof T>(event?: K): void {
    if (event) {
      this.handlers.delete(event)
    } else {
      this.handlers.clear()
    }
  }

  listenerCount<K extends keyof T>(event: K): number {
    return this.handlers.get(event)?.size || 0
  }
}

/**
 * Hook para crear un emisor de eventos
 */
export function useEventEmitter<T extends Record<string, any> = Record<string, any>>() {
  const emitterRef = useRef<EventEmitter<T>>(new EventEmitter<T>())

  const on = useCallback(<K extends keyof T>(
    event: K,
    handler: EventHandler<T[K]>
  ) => {
    return emitterRef.current.on(event, handler)
  }, [])

  const off = useCallback(<K extends keyof T>(
    event: K,
    handler: EventHandler<T[K]>
  ) => {
    emitterRef.current.off(event, handler)
  }, [])

  const emit = useCallback(<K extends keyof T>(
    event: K,
    data: T[K]
  ) => {
    emitterRef.current.emit(event, data)
  }, [])

  const once = useCallback(<K extends keyof T>(
    event: K,
    handler: EventHandler<T[K]>
  ) => {
    return emitterRef.current.once(event, handler)
  }, [])

  const removeAllListeners = useCallback(<K extends keyof T>(event?: K) => {
    emitterRef.current.removeAllListeners(event)
  }, [])

  const listenerCount = useCallback(<K extends keyof T>(event: K) => {
    return emitterRef.current.listenerCount(event)
  }, [])

  // Limpiar al desmontar
  useEffect(() => {
    return () => {
      emitterRef.current.removeAllListeners()
    }
  }, [])

  return {
    on,
    off,
    emit,
    once,
    removeAllListeners,
    listenerCount
  }
}






