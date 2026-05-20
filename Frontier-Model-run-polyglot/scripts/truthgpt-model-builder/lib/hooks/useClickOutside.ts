/**
 * Hook useClickOutside
 * ====================
 * 
 * Hook para detectar clicks fuera de un elemento
 */

import { useEffect, useRef, RefObject } from 'react'

export interface UseClickOutsideOptions {
  enabled?: boolean
  handler: (event: MouseEvent | TouchEvent) => void
}

/**
 * Hook para detectar clicks fuera de un elemento
 */
export function useClickOutside<T extends HTMLElement = HTMLElement>(
  options: UseClickOutsideOptions
): RefObject<T> {
  const { enabled = true, handler } = options
  const ref = useRef<T>(null)

  useEffect(() => {
    if (!enabled) return

    const handleClickOutside = (event: MouseEvent | TouchEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        handler(event)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('touchstart', handleClickOutside)

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('touchstart', handleClickOutside)
    }
  }, [enabled, handler])

  return ref
}







