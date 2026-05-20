/**
 * Theme Context Provider
 * Provides global theme state
 */

'use client'

import React, { createContext, useContext, ReactNode } from 'react'
import { useTheming } from '../hooks/useTheming'

interface ThemeContextType extends ReturnType<typeof useTheming> {}

const ThemeContext = createContext<ThemeContextType | null>(null)

interface ThemeProviderProps {
  children: ReactNode
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const theming = useTheming()

  return (
    <ThemeContext.Provider value={theming}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}




