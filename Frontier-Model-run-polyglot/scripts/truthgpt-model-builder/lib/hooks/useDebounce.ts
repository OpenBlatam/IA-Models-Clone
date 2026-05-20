/**
 * Hook useDebounce Mejorado
 * ==========================
 * 
 * Hook para debounce con cancelación y estado
 */

import { useState, useEffect, useRef, useCallback } from 'react'

export interface UseDebounceOptions {
  delay?: number
  immediate?: boolean
}

export interface UseDebounceResult<T> {
  debouncedValue: T
  isDebouncing: boolean
  cancel: () => void
  flush: () => void
}

/**
 * Hook para debounce de valores
 */
export function useDebounce<T>(
  value: T,
  options: UseDebounceOptions = {}
): UseDebounceResult<T> {
  const { delay = 500, immediate = false } = options
  const [debouncedValue, setDebouncedValue] = useState<T>(value)
  const [isDebouncing, setIsDebouncing] = useState(false)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)
  const immediateRef = useRef(immediate)

  useEffect(() => {
    immediateRef.current = immediate
  }, [immediate])

  const cancel = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
      setIsDebouncing(false)
    }
  }, [])

  const flush = useCallback(() => {
    cancel()
    setDebouncedValue(value)
    setIsDebouncing(false)
  }, [value, cancel])

  useEffect(() => {
    if (immediateRef.current && debouncedValue !== value) {
      setDebouncedValue(value)
      setIsDebouncing(false)
      return
    }

    setIsDebouncing(true)

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    timeoutRef.current = setTimeout(() => {
      setDebouncedValue(value)
      setIsDebouncing(false)
      timeoutRef.current = null
    }, delay)

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [value, delay])

  return {
    debouncedValue,
    isDebouncing,
    cancel,
    flush
  }
}

/**
 * Hook para debounce de funciones
 */
export function useDebouncedCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number = 500
): [T & { cancel: () => void; flush: () => void }, boolean] {
  const [isDebouncing, setIsDebouncing] = useState(false)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)
  const callbackRef = useRef(callback)
  const argsRef = useRef<Parameters<T>>()

  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  const debouncedCallback = useCallback(
    ((...args: Parameters<T>) => {
      argsRef.current = args
      setIsDebouncing(true)

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }

      timeoutRef.current = setTimeout(() => {
        callbackRef.current(...argsRef.current!)
        setIsDebouncing(false)
        timeoutRef.current = null
      }, delay)
    }) as T & { cancel: () => void; flush: () => void },
    [delay]
  )

  debouncedCallback.cancel = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
      setIsDebouncing(false)
    }
  }, [])

  debouncedCallback.flush = useCallback(() => {
    if (timeoutRef.current && argsRef.current) {
      clearTimeout(timeoutRef.current)
      callbackRef.current(...argsRef.current)
      timeoutRef.current = null
      setIsDebouncing(false)
    }
  }, [])

  return [debouncedCallback, isDebouncing]
}
