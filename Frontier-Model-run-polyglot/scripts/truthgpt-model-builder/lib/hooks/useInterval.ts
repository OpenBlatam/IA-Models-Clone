/**
 * Hook useInterval
 * ================
 * 
 * Hook para ejecutar una función en intervalos
 */

import { useEffect, useRef } from 'react'

export interface UseIntervalOptions {
  immediate?: boolean // Ejecutar inmediatamente
  enabled?: boolean // Habilitar/deshabilitar el intervalo
}

/**
 * Hook para ejecutar una función en intervalos
 */
export function useInterval(
  callback: () => void,
  delay: number | null,
  options: UseIntervalOptions = {}
): void {
  const { immediate = false, enabled = true } = options
  const savedCallback = useRef(callback)

  // Actualizar callback ref cuando cambia
  useEffect(() => {
    savedCallback.current = callback
  }, [callback])

  useEffect(() => {
    if (!enabled || delay === null) return

    // Ejecutar inmediatamente si está habilitado
    if (immediate) {
      savedCallback.current()
    }

    const interval = setInterval(() => {
      savedCallback.current()
    }, delay)

    return () => clearInterval(interval)
  }, [delay, enabled, immediate])
}






