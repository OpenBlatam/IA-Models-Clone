/**
 * Hook useWindowSize
 * ==================
 * 
 * Hook para obtener el tamaño de la ventana
 */

import { useState, useEffect } from 'react'

export interface WindowSize {
  width: number
  height: number
}

/**
 * Hook para obtener el tamaño de la ventana
 */
export function useWindowSize(): WindowSize {
  const [windowSize, setWindowSize] = useState<WindowSize>(() => {
    if (typeof window === 'undefined') {
      return { width: 0, height: 0 }
    }
    return {
      width: window.innerWidth,
      height: window.innerHeight
    }
  })

  useEffect(() => {
    if (typeof window === 'undefined') return

    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      })
    }

    window.addEventListener('resize', handleResize)
    handleResize() // Llamar una vez para establecer el tamaño inicial

    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return windowSize
}







