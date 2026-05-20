/**
 * Hook useId
 * ==========
 * 
 * Hook para generar IDs únicos
 */

import { useRef } from 'react'

let idCounter = 0

/**
 * Genera un ID único
 */
function generateId(prefix: string = 'id'): string {
  return `${prefix}-${++idCounter}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Hook para obtener un ID único estable
 */
export function useId(prefix: string = 'id'): string {
  const idRef = useRef<string>()

  if (!idRef.current) {
    idRef.current = generateId(prefix)
  }

  return idRef.current
}

/**
 * Hook para generar múltiples IDs
 */
export function useIds(count: number, prefix: string = 'id'): string[] {
  const idsRef = useRef<string[]>()

  if (!idsRef.current) {
    idsRef.current = Array.from({ length: count }, () => generateId(prefix))
  }

  return idsRef.current
}






