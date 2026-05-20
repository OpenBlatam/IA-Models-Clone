/**
 * Hook usePersistentState
 * =======================
 * 
 * Hook para estado persistente en localStorage
 */

import { useState, useEffect, useCallback } from 'react'
import { getLocalStorage, setLocalStorage } from '../utils/storageUtils'

export interface UsePersistentStateOptions {
  key: string
  defaultValue?: any
  serialize?: (value: any) => string
  deserialize?: (value: string) => any
  sync?: boolean // Sincronizar entre tabs
}

/**
 * Hook para estado persistente en localStorage
 */
export function usePersistentState<T>(
  options: UsePersistentStateOptions
): [T, (value: T | ((prev: T) => T)) => void] {
  const {
    key,
    defaultValue,
    serialize = JSON.stringify,
    deserialize = JSON.parse,
    sync = true
  } = options

  const [state, setState] = useState<T>(() => {
    const stored = getLocalStorage<string>(key)
    if (stored === null) return defaultValue
    try {
      return deserialize(stored)
    } catch {
      return defaultValue
    }
  })

  // Guardar en localStorage cuando cambia el estado
  useEffect(() => {
    try {
      const serialized = serialize(state)
      setLocalStorage(key, serialized)
    } catch (error) {
      console.error('Error al guardar estado persistente:', error)
    }
  }, [state, key, serialize])

  // Sincronizar entre tabs
  useEffect(() => {
    if (!sync) return

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try {
          const newState = deserialize(e.newValue)
          setState(newState)
        } catch (error) {
          console.error('Error al deserializar estado:', error)
        }
      }
    }

    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [key, deserialize, sync])

  const updateState = useCallback((value: T | ((prev: T) => T)) => {
    setState(prev => {
      const newValue = typeof value === 'function' 
        ? (value as (prev: T) => T)(prev)
        : value
      return newValue
    })
  }, [])

  return [state, updateState]
}






