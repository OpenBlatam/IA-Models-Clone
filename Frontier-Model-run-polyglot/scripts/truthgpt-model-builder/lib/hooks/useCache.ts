/**
 * Hook useCache
 * =============
 * 
 * Hook para manejar cache en componentes
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { LRUCache } from '../utils/cacheUtils'

export interface UseCacheOptions {
  maxSize?: number
  defaultTTL?: number | null
}

/**
 * Hook para manejar cache en componentes
 */
export function useCache<T = any>(options: UseCacheOptions = {}) {
  const { maxSize = 100, defaultTTL = null } = options
  const cacheRef = useRef(new LRUCache<T>(maxSize, defaultTTL))
  const [, forceUpdate] = useState(0)

  // Limpiar elementos expirados periódicamente
  useEffect(() => {
    const interval = setInterval(() => {
      cacheRef.current.cleanExpired()
    }, 60000) // Cada minuto

    return () => clearInterval(interval)
  }, [])

  const get = useCallback((key: string): T | null => {
    return cacheRef.current.get(key)
  }, [])

  const set = useCallback((key: string, value: T, ttl?: number | null): void => {
    cacheRef.current.set(key, value, ttl)
    forceUpdate(prev => prev + 1)
  }, [])

  const remove = useCallback((key: string): boolean => {
    const result = cacheRef.current.delete(key)
    forceUpdate(prev => prev + 1)
    return result
  }, [])

  const clear = useCallback((): void => {
    cacheRef.current.clear()
    forceUpdate(prev => prev + 1)
  }, [])

  const has = useCallback((key: string): boolean => {
    return cacheRef.current.has(key)
  }, [])

  const size = useCallback((): number => {
    return cacheRef.current.size()
  }, [])

  const keys = useCallback((): string[] => {
    return cacheRef.current.keys()
  }, [])

  const stats = useCallback(() => {
    return cacheRef.current.getStats()
  }, [])

  return {
    get,
    set,
    remove,
    clear,
    has,
    size,
    keys,
    stats
  }
}






