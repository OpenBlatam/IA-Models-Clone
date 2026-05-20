/**
 * Hook useKeyboard
 * ================
 * 
 * Hook para manejar eventos de teclado
 */

import { useEffect, useCallback, useState, RefObject } from 'react'

export type KeyCode = string | number
export type KeyboardHandler = (event: KeyboardEvent) => void

export interface UseKeyboardOptions {
  target?: RefObject<HTMLElement> | HTMLElement | null
  enabled?: boolean
  preventDefault?: boolean
  stopPropagation?: boolean
}

export interface UseKeyboardResult {
  onKeyDown: (handler: KeyboardHandler) => void
  onKeyUp: (handler: KeyboardHandler) => void
  onKeyPress: (handler: KeyboardHandler) => void
}

/**
 * Hook para manejar eventos de teclado
 */
export function useKeyboard(
  key: KeyCode | KeyCode[],
  handler: KeyboardHandler,
  options: UseKeyboardOptions = {}
): void {
  const {
    target,
    enabled = true,
    preventDefault = false,
    stopPropagation = false
  } = options

  const keys = Array.isArray(key) ? key : [key]

  useEffect(() => {
    if (!enabled) return

    const handleKeyDown = (event: KeyboardEvent) => {
      const keyCode = event.key || event.code || event.keyCode
      const keyString = String(keyCode)

      if (
        keys.some(k => 
          String(k) === keyString || 
          String(k) === event.key ||
          String(k) === event.code
        )
      ) {
        if (preventDefault) event.preventDefault()
        if (stopPropagation) event.stopPropagation()
        handler(event)
      }
    }

    const targetElement = target
      ? ('current' in target ? target.current : target)
      : window

    if (targetElement) {
      targetElement.addEventListener('keydown', handleKeyDown)
      return () => {
        targetElement.removeEventListener('keydown', handleKeyDown)
      }
    }
  }, [key, handler, enabled, preventDefault, stopPropagation, target])
}

/**
 * Hook para manejar múltiples combinaciones de teclas
 */
export function useKeyboardShortcuts(
  shortcuts: Record<string, KeyboardHandler>,
  options: UseKeyboardOptions = {}
): void {
  const {
    target,
    enabled = true,
    preventDefault = true,
    stopPropagation = true
  } = options

  useEffect(() => {
    if (!enabled) return

    const handleKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase()
      const ctrl = event.ctrlKey || event.metaKey
      const shift = event.shiftKey
      const alt = event.altKey

      // Construir combinación de teclas
      const parts: string[] = []
      if (ctrl) parts.push('ctrl')
      if (shift) parts.push('shift')
      if (alt) parts.push('alt')
      parts.push(key)

      const combination = parts.join('+')

      if (shortcuts[combination]) {
        if (preventDefault) event.preventDefault()
        if (stopPropagation) event.stopPropagation()
        shortcuts[combination](event)
      }
    }

    const targetElement = target
      ? ('current' in target ? target.current : target)
      : window

    if (targetElement) {
      targetElement.addEventListener('keydown', handleKeyDown)
      return () => {
        targetElement.removeEventListener('keydown', handleKeyDown)
      }
    }
  }, [shortcuts, enabled, preventDefault, stopPropagation, target])
}

/**
 * Hook para detectar si una tecla está presionada
 */
export function useKeyPress(key: KeyCode): boolean {
  const [isPressed, setIsPressed] = useState(false)

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const keyCode = event.key || event.code || event.keyCode
      if (String(keyCode) === String(key)) {
        setIsPressed(true)
      }
    }

    const handleKeyUp = (event: KeyboardEvent) => {
      const keyCode = event.key || event.code || event.keyCode
      if (String(keyCode) === String(key)) {
        setIsPressed(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [key])

  return isPressed
}

