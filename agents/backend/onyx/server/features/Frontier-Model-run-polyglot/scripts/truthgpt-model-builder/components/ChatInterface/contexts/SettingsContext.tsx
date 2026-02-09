/**
 * Settings Context Provider
 * Provides global settings state
 */

'use client'

import React, { createContext, useContext, ReactNode } from 'react'
import { useSettings } from '../hooks/useSettings'

interface SettingsContextType extends ReturnType<typeof useSettings> {}

const SettingsContext = createContext<SettingsContextType | null>(null)

interface SettingsProviderProps {
  children: ReactNode
}

export function SettingsProvider({ children }: SettingsProviderProps) {
  const settings = useSettings()

  return (
    <SettingsContext.Provider value={settings}>
      {children}
    </SettingsContext.Provider>
  )
}

export function useSettingsContext(): SettingsContextType {
  const context = useContext(SettingsContext)
  if (!context) {
    throw new Error('useSettingsContext must be used within SettingsProvider')
  }
  return context
}




