/**
 * Hook usePrevious
 * ================
 * 
 * Hook para obtener el valor anterior de una variable
 */

import { useRef, useEffect } from 'react'

/**
 * Hook para obtener el valor anterior
 */
export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>()

  useEffect(() => {
    ref.current = value
  }, [value])

  return ref.current
}

/**
 * Hook para obtener múltiples valores anteriores
 */
export function usePreviousValues<T>(value: T, count: number = 5): T[] {
  const ref = useRef<T[]>([])

  useEffect(() => {
    ref.current = [value, ...ref.current].slice(0, count)
  }, [value, count])

  return ref.current
}
