import { useState, useCallback } from 'react'

/**
 * Generic hook for managing Map-based state
 * Eliminates duplicate code for Map state management
 */
export function useMapState<T>(initialValue: Map<string, T> = new Map()) {
  const [map, setMap] = useState<Map<string, T>>(initialValue)

  const set = useCallback((key: string, value: T) => {
    setMap((prev) => {
      const newMap = new Map(prev)
      newMap.set(key, value)
      return newMap
    })
  }, [])

  const get = useCallback((key: string): T | undefined => {
    return map.get(key)
  }, [map])

  const has = useCallback((key: string): boolean => {
    return map.has(key)
  }, [map])

  const remove = useCallback((key: string) => {
    setMap((prev) => {
      const newMap = new Map(prev)
      newMap.delete(key)
      return newMap
    })
  }, [])

  const clear = useCallback(() => {
    setMap(new Map())
  }, [])

  const update = useCallback((key: string, updater: (value: T | undefined) => T) => {
    setMap((prev) => {
      const newMap = new Map(prev)
      const current = newMap.get(key)
      newMap.set(key, updater(current))
      return newMap
    })
  }, [])

  const setAll = useCallback((newMap: Map<string, T>) => {
    setMap(newMap)
  }, [])

  const size = map.size
  const entries = Array.from(map.entries())
  const keys = Array.from(map.keys())
  const values = Array.from(map.values())

  return {
    map,
    set,
    get,
    has,
    remove,
    clear,
    update,
    setAll,
    size,
    entries,
    keys,
    values,
  }
}




