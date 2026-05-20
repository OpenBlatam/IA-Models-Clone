/**
 * Hook para mejoras de accesibilidad
 * ===================================
 */

import { useEffect, useCallback, useState, useRef } from 'react'

export interface AccessibilityOptions {
  enableAnnouncements?: boolean
  enableKeyboardNavigation?: boolean
  announceErrors?: boolean
  announceSuccess?: boolean
  ariaLabels?: Record<string, string>
}

export interface UseModelAccessibilityResult {
  announce: (message: string, priority?: 'polite' | 'assertive') => void
  announceError: (error: string) => void
  announceSuccess: (message: string) => void
  setAriaLabel: (id: string, label: string) => void
  getAriaLabel: (id: string) => string
  focusElement: (id: string) => void
  isKeyboardNavigation: boolean
}

/**
 * Hook para mejoras de accesibilidad
 */
export function useModelAccessibility(
  options: AccessibilityOptions = {}
): UseModelAccessibilityResult {
  const {
    enableAnnouncements = true,
    enableKeyboardNavigation = true,
    announceErrors = true,
    announceSuccess = true,
    ariaLabels: initialAriaLabels = {}
  } = options

  const [ariaLabels, setAriaLabels] = useState<Record<string, string>>(initialAriaLabels)
  const [isKeyboardNavigation, setIsKeyboardNavigation] = useState(false)
  const announcementRef = useRef<HTMLDivElement | null>(null)

  // Crear elemento para anuncios screen reader
  useEffect(() => {
    if (enableAnnouncements && typeof window !== 'undefined') {
      let element = document.getElementById('model-announcements')
      
      if (!element) {
        element = document.createElement('div')
        element.id = 'model-announcements'
        element.setAttribute('role', 'status')
        element.setAttribute('aria-live', 'polite')
        element.setAttribute('aria-atomic', 'true')
        element.className = 'sr-only'
        element.style.cssText = 'position: absolute; left: -10000px; width: 1px; height: 1px; overflow: hidden;'
        document.body.appendChild(element)
      }

      announcementRef.current = element
    }
  }, [enableAnnouncements])

  // Detectar navegación por teclado
  useEffect(() => {
    if (!enableKeyboardNavigation) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        setIsKeyboardNavigation(true)
      }
    }

    const handleMouseDown = () => {
      setIsKeyboardNavigation(false)
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('mousedown', handleMouseDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('mousedown', handleMouseDown)
    }
  }, [enableKeyboardNavigation])

  const announce = useCallback((
    message: string,
    priority: 'polite' | 'assertive' = 'polite'
  ) => {
    if (!enableAnnouncements || !announcementRef.current) return

    const element = announcementRef.current
    element.setAttribute('aria-live', priority)
    element.textContent = message

    // Limpiar después de un momento para que el mensaje pueda ser anunciado de nuevo
    setTimeout(() => {
      if (element) {
        element.textContent = ''
      }
    }, 1000)
  }, [enableAnnouncements])

  const setAriaLabel = useCallback((id: string, label: string) => {
    setAriaLabels(prev => ({ ...prev, [id]: label }))
    
    // Actualizar elemento si existe
    if (typeof document !== 'undefined') {
      const element = document.getElementById(id)
      if (element) {
        element.setAttribute('aria-label', label)
      }
    }
  }, [])

  const getAriaLabel = useCallback((id: string) => {
    return ariaLabels[id] || ''
  }, [ariaLabels])

  const focusElement = useCallback((id: string) => {
    if (typeof document !== 'undefined') {
      const element = document.getElementById(id)
      if (element) {
        element.focus()
      }
    }
  }, [])

  // Annunciar errores automáticamente
  const announceError = useCallback((error: string) => {
    if (announceErrors) {
      announce(`Error: ${error}`, 'assertive')
    }
  }, [announceErrors, announce])

  // Annunciar éxito automáticamente
  const announceSuccess = useCallback((message: string) => {
    if (announceSuccess) {
      announce(`Éxito: ${message}`, 'polite')
    }
  }, [announceSuccess, announce])

  return {
    announce,
    announceError,
    announceSuccess,
    setAriaLabel,
    getAriaLabel,
    focusElement,
    isKeyboardNavigation
  }
}

