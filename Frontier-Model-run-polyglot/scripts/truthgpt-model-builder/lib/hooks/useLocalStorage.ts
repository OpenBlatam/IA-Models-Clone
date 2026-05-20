/**
 * Hook para LocalStorage Mejorado
 * ================================
 * 
 * Hook para usar localStorage con type safety y sincronización
 */

import { useState, useEffect, useCallback } from 'react'

export interface UseLocalStorageOptions<T> {
  defaultValue?: T
  serializer?: (value: T) => string
  deserializer?: (value: string) => T
  sync?: boolean // Sincronizar entre tabs
}

export interface UseLocalStorageResult<T> {
  value: T
  setValue: (value: T | ((prev: T) => T)) => void
  removeValue: () => void
  isLoaded: boolean
}

/**
 * Hook para usar localStorage con type safety
 */
export function useLocalStorage<T>(
  key: string,
  options: UseLocalStorageOptions<T> = {}
): UseLocalStorageResult<T> {
  const {
    defaultValue,
    serializer = JSON.stringify,
    deserializer = JSON.parse,
    sync = true
  } = options

  const [value, setValueState] = useState<T>(() => {
    if (typeof window === 'undefined') {
      return defaultValue as T
    }

    try {
      const item = window.localStorage.getItem(key)
      if (item === null) {
        return defaultValue as T
      }
      return deserializer(item)
    } catch {
      return defaultValue as T
    }
  })

  const [isLoaded, setIsLoaded] = useState(false)

  // Cargar valor inicial
  useEffect(() => {
    if (typeof window === 'undefined') return

    try {
      const item = window.localStorage.getItem(key)
      if (item !== null) {
        setValueState(deserializer(item))
      }
      setIsLoaded(true)
    } catch {
      setIsLoaded(true)
    }
  }, [key, deserializer])

  // Sincronizar entre tabs
  useEffect(() => {
    if (!sync || typeof window === 'undefined') return

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try {
          setValueState(deserializer(e.newValue))
        } catch {
          // Ignorar errores de deserialización
        }
      }
    }

    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [key, deserializer, sync])

  const setValue = useCallback((newValue: T | ((prev: T) => T)) => {
    try {
      const valueToStore = newValue instanceof Function 
        ? newValue(value) 
        : newValue

      setValueState(valueToStore)

      if (typeof window !== 'undefined') {
        window.localStorage.setItem(key, serializer(valueToStore))
      }
    } catch (error) {
      console.error(`Error saving to localStorage key "${key}":`, error)
    }
  }, [key, serializer, value])

  const removeValue = useCallback(() => {
    try {
      setValueState(defaultValue as T)
      if (typeof window !== 'undefined') {
        window.localStorage.removeItem(key)
      }
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error)
    }
  }, [key, defaultValue])

  return {
    value,
    setValue,
    removeValue,
    isLoaded
  }
}
