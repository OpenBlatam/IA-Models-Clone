/**
 * Hook para caché inteligente de modelos
 * ========================================
 */

import { useState, useCallback, useRef, useEffect } from 'react'

export interface CachedModel {
  id: string
  name: string
  description: string
  spec: any
  createdAt: number
  lastAccessed: number
  accessCount: number
}

export interface UseModelCacheOptions {
  maxSize?: number
  ttl?: number // Time to live en milisegundos
  enableAutoCleanup?: boolean
}

export interface UseModelCacheResult {
  get: (description: string) => CachedModel | null
  set: (description: string, model: Omit<CachedModel, 'createdAt' | 'lastAccessed' | 'accessCount'>) => void
  clear: () => void
  size: number
  stats: {
    hits: number
    misses: number
    hitRate: number
  }
}

/**
 * Genera una clave de caché basada en la descripción
 */
function generateCacheKey(description: string): string {
  return description.trim().toLowerCase().replace(/\s+/g, '-')
}

/**
 * Hook para caché de modelos con LRU y TTL
 */
export function useModelCache(options: UseModelCacheOptions = {}): UseModelCacheResult {
  const {
    maxSize = 50,
    ttl = 24 * 60 * 60 * 1000, // 24 horas por defecto
    enableAutoCleanup = true
  } = options

  const [cache, setCache] = useState<Map<string, CachedModel>>(new Map())
  const statsRef = useRef({ hits: 0, misses: 0 })

  // Limpiar elementos expirados
  const cleanupExpired = useCallback(() => {
    const now = Date.now()
    setCache(prev => {
      const next = new Map(prev)
      let removed = 0

      for (const [key, value] of next.entries()) {
        if (now - value.lastAccessed > ttl) {
          next.delete(key)
          removed++
        }
      }

      return next
    })
  }, [ttl])

  // Limpiar elementos más antiguos si excede el tamaño máximo
  const enforceMaxSize = useCallback(() => {
    setCache(prev => {
      if (prev.size <= maxSize) return prev

      const next = new Map(prev)
      const entries = Array.from(next.entries())
      
      // Ordenar por último acceso (LRU)
      entries.sort((a, b) => a[1].lastAccessed - b[1].lastAccessed)

      // Eliminar los más antiguos
      const toRemove = entries.slice(0, entries.length - maxSize)
      toRemove.forEach(([key]) => next.delete(key))

      return next
    })
  }, [maxSize])

  // Auto-limpieza periódica
  useEffect(() => {
    if (!enableAutoCleanup) return

    const interval = setInterval(() => {
      cleanupExpired()
      enforceMaxSize()
    }, 60 * 1000) // Cada minuto

    return () => clearInterval(interval)
  }, [cleanupExpired, enforceMaxSize, enableAutoCleanup])

  const get = useCallback((description: string): CachedModel | null => {
    const key = generateCacheKey(description)
    const cached = cache.get(key)

    if (!cached) {
      statsRef.current.misses++
      return null
    }

    // Verificar si está expirado
    const now = Date.now()
    if (now - cached.lastAccessed > ttl) {
      setCache(prev => {
        const next = new Map(prev)
        next.delete(key)
        return next
      })
      statsRef.current.misses++
      return null
    }

    // Actualizar último acceso y contador
    setCache(prev => {
      const next = new Map(prev)
      next.set(key, {
        ...cached,
        lastAccessed: now,
        accessCount: cached.accessCount + 1
      })
      return next
    })

    statsRef.current.hits++
    return cached
  }, [cache, ttl])

  const set = useCallback((
    description: string,
    model: Omit<CachedModel, 'createdAt' | 'lastAccessed' | 'accessCount'>
  ) => {
    const key = generateCacheKey(description)
    const now = Date.now()

    setCache(prev => {
      const next = new Map(prev)
      
      // Verificar si ya existe
      const existing = next.get(key)
      const accessCount = existing?.accessCount || 0

      next.set(key, {
        ...model,
        createdAt: existing?.createdAt || now,
        lastAccessed: now,
        accessCount: accessCount + 1
      })

      // Limpiar si excede el tamaño máximo
      if (next.size > maxSize) {
        const entries = Array.from(next.entries())
        entries.sort((a, b) => a[1].lastAccessed - b[1].lastAccessed)
        const toRemove = entries.slice(0, entries.length - maxSize)
        toRemove.forEach(([k]) => next.delete(k))
      }

      return next
    })
  }, [maxSize])

  const clear = useCallback(() => {
    setCache(new Map())
    statsRef.current = { hits: 0, misses: 0 }
  }, [])

  const stats = {
    hits: statsRef.current.hits,
    misses: statsRef.current.misses,
    hitRate: statsRef.current.hits + statsRef.current.misses > 0
      ? statsRef.current.hits / (statsRef.current.hits + statsRef.current.misses)
      : 0
  }

  return {
    get,
    set,
    clear,
    size: cache.size,
    stats
  }
}

