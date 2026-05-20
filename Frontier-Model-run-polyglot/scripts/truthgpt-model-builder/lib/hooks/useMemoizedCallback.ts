/**
 * Hook useMemoizedCallback
 * ========================
 * 
 * Hook para memoizar callbacks con dependencias
 */

import { useCallback, useRef } from 'react'

/**
 * Hook para memoizar callbacks con comparación profunda de dependencias
 */
export function useMemoizedCallback<T extends (...args: any[]) => any>(
  callback: T,
  deps: any[]
): T {
  const callbackRef = useRef(callback)
  const depsRef = useRef(deps)

  // Actualizar callback
  callbackRef.current = callback

  // Verificar si las dependencias cambiaron
  const depsChanged = depsRef.current.length !== deps.length ||
    depsRef.current.some((dep, index) => dep !== deps[index])

  if (depsChanged) {
    depsRef.current = deps
  }

  return useCallback(
    ((...args: any[]) => callbackRef.current(...args)) as T,
    depsChanged ? deps : depsRef.current
  )
}

/**
 * Hook para memoizar callbacks sin dependencias (siempre el mismo)
 */
export function useStableCallback<T extends (...args: any[]) => any>(
  callback: T
): T {
  const callbackRef = useRef(callback)

  callbackRef.current = callback

  return useCallback(
    ((...args: any[]) => callbackRef.current(...args)) as T,
    []
  )
}






