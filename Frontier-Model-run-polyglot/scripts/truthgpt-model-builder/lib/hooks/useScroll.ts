/**
 * Hook useScroll
 * ==============
 * 
 * Hook para detectar scroll
 */

import { useState, useEffect } from 'react'

export interface ScrollPosition {
  x: number
  y: number
}

export interface UseScrollOptions {
  element?: HTMLElement | null
  throttle?: number
}

/**
 * Hook para detectar posición de scroll
 */
export function useScroll(options: UseScrollOptions = {}): ScrollPosition {
  const { element, throttle = 100 } = options
  const [scrollPosition, setScrollPosition] = useState<ScrollPosition>({ x: 0, y: 0 })

  useEffect(() => {
    const target = element || window
    let timeoutId: NodeJS.Timeout | null = null

    const handleScroll = () => {
      if (timeoutId) return

      timeoutId = setTimeout(() => {
        if (element) {
          setScrollPosition({
            x: element.scrollLeft,
            y: element.scrollTop
          })
        } else {
          setScrollPosition({
            x: window.scrollX || window.pageXOffset,
            y: window.scrollY || window.pageYOffset
          })
        }
        timeoutId = null
      }, throttle)
    }

    target.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll() // Llamar una vez para establecer posición inicial

    return () => {
      target.removeEventListener('scroll', handleScroll)
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [element, throttle])

  return scrollPosition
}

/**
 * Hook para detectar si se ha hecho scroll
 */
export function useScrolled(threshold: number = 0): boolean {
  const { y } = useScroll()
  return y > threshold
}







