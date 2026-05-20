/**
 * Hook useTimeout
 * ===============
 * 
 * Hook para ejecutar una función después de un delay
 */

import { useEffect, useRef } from 'react'

export interface UseTimeoutOptions {
  enabled?: boolean // Habilitar/deshabilitar el timeout
}

/**
 * Hook para ejecutar una función después de un delay
 */
export function useTimeout(
  callback: () => void,
  delay: number | null,
  options: UseTimeoutOptions = {}
): void {
  const { enabled = true } = options
  const savedCallback = useRef(callback)

  // Actualizar callback ref cuando cambia
  useEffect(() => {
    savedCallback.current = callback
  }, [callback])

  useEffect(() => {
    if (!enabled || delay === null) return

    const timeout = setTimeout(() => {
      savedCallback.current()
    }, delay)

    return () => clearTimeout(timeout)
  }, [delay, enabled])
}






