import { useState, useCallback } from 'react'

export interface AccessibilityState {
  screenReader: boolean
  highContrast: boolean
  activeTheme: string
  fontSize: 'small' | 'medium' | 'large'
}

export interface AccessibilityActions {
  setScreenReader: (enabled: boolean) => void
  setHighContrast: (enabled: boolean) => void
  setActiveTheme: (theme: string) => void
  setFontSize: (size: 'small' | 'medium' | 'large') => void
}

export function useAccessibilityState(initialFontSize: 'small' | 'medium' | 'large' = 'medium') {
  const [screenReader, setScreenReader] = useState(false)
  const [highContrast, setHighContrast] = useState(false)
  const [activeTheme, setActiveTheme] = useState<string>('default')
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large'>(initialFontSize)

  const state: AccessibilityState = {
    screenReader,
    highContrast,
    activeTheme,
    fontSize
  }

  const actions: AccessibilityActions = {
    setScreenReader: useCallback((enabled: boolean) => setScreenReader(enabled), []),
    setHighContrast: useCallback((enabled: boolean) => setHighContrast(enabled), []),
    setActiveTheme: useCallback((theme: string) => setActiveTheme(theme), []),
    setFontSize: useCallback((size: 'small' | 'medium' | 'large') => setFontSize(size), [])
  }

  return { state, actions }
}



