/**
 * Custom hook for theming and customization
 * Handles themes, custom themes, colors, and visual customization
 */

import { useState, useCallback, useEffect } from 'react'

export interface ThemeState {
  theme: 'dark' | 'light' | 'auto'
  customThemes: Map<string, any>
  activeTheme: string
  customThemeColors: Map<string, string>
  themePresets: Map<string, any>
}

export interface ThemeActions {
  setTheme: (theme: 'dark' | 'light' | 'auto') => void
  setActiveTheme: (themeName: string) => void
  addCustomTheme: (name: string, theme: any) => void
  removeCustomTheme: (name: string) => void
  updateCustomTheme: (name: string, updates: any) => void
  setCustomColor: (key: string, color: string) => void
  addThemePreset: (name: string, preset: any) => void
  removeThemePreset: (name: string) => void
  applyTheme: (themeName: string) => void
  exportTheme: (themeName: string) => string
  importTheme: (themeJson: string) => void
}

const STORAGE_KEY = 'chat-theme-settings'

const DEFAULT_THEMES = {
  dark: {
    background: '#1a1a1a',
    foreground: '#ffffff',
    primary: '#007bff',
    secondary: '#6c757d',
    accent: '#28a745',
    border: '#333333',
  },
  light: {
    background: '#ffffff',
    foreground: '#000000',
    primary: '#007bff',
    secondary: '#6c757d',
    accent: '#28a745',
    border: '#e0e0e0',
  },
}

export function useTheming(): ThemeState & ThemeActions {
  const [theme, setTheme] = useState<'dark' | 'light' | 'auto'>('dark')
  const [customThemes, setCustomThemes] = useState<Map<string, any>>(new Map())
  const [activeTheme, setActiveTheme] = useState<string>('default')
  const [customThemeColors, setCustomThemeColors] = useState<Map<string, string>>(new Map())
  const [themePresets, setThemePresets] = useState<Map<string, any>>(new Map())

  // Load from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        setTheme(parsed.theme || 'dark')
        setCustomThemes(new Map(parsed.customThemes || []))
        setActiveTheme(parsed.activeTheme || 'default')
        setCustomThemeColors(new Map(parsed.customThemeColors || []))
        setThemePresets(new Map(parsed.themePresets || []))
      }
    } catch (error) {
      console.error('Error loading theme settings:', error)
    }
  }, [])

  // Save to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        theme,
        customThemes: Array.from(customThemes.entries()),
        activeTheme,
        customThemeColors: Array.from(customThemeColors.entries()),
        themePresets: Array.from(themePresets.entries()),
      }))
    } catch (error) {
      console.error('Error saving theme settings:', error)
    }
  }, [theme, customThemes, activeTheme, customThemeColors, themePresets])

  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement
    let themeToApply = theme

    if (theme === 'auto') {
      themeToApply = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }

    const themeConfig = activeTheme === 'default' 
      ? DEFAULT_THEMES[themeToApply]
      : customThemes.get(activeTheme) || DEFAULT_THEMES[themeToApply]

    // Apply CSS variables
    Object.entries(themeConfig).forEach(([key, value]) => {
      root.style.setProperty(`--theme-${key}`, value as string)
    })

    // Apply custom colors
    customThemeColors.forEach((color, key) => {
      root.style.setProperty(`--custom-${key}`, color)
    })

    // Update class
    root.classList.remove('theme-dark', 'theme-light')
    root.classList.add(`theme-${themeToApply}`)
  }, [theme, activeTheme, customThemes, customThemeColors])

  const addCustomTheme = useCallback((name: string, themeConfig: any) => {
    setCustomThemes(prev => {
      const next = new Map(prev)
      next.set(name, themeConfig)
      return next
    })
  }, [])

  const removeCustomTheme = useCallback((name: string) => {
    setCustomThemes(prev => {
      const next = new Map(prev)
      next.delete(name)
      return next
    })
  }, [])

  const updateCustomTheme = useCallback((name: string, updates: any) => {
    setCustomThemes(prev => {
      const next = new Map(prev)
      const existing = next.get(name)
      if (existing) {
        next.set(name, { ...existing, ...updates })
      }
      return next
    })
  }, [])

  const setCustomColor = useCallback((key: string, color: string) => {
    setCustomThemeColors(prev => {
      const next = new Map(prev)
      next.set(key, color)
      return next
    })
  }, [])

  const addThemePreset = useCallback((name: string, preset: any) => {
    setThemePresets(prev => {
      const next = new Map(prev)
      next.set(name, preset)
      return next
    })
  }, [])

  const removeThemePreset = useCallback((name: string) => {
    setThemePresets(prev => {
      const next = new Map(prev)
      next.delete(name)
      return next
    })
  }, [])

  const applyTheme = useCallback((themeName: string) => {
    setActiveTheme(themeName)
  }, [])

  const exportTheme = useCallback((themeName: string): string => {
    const themeToExport = customThemes.get(themeName)
    if (!themeToExport) {
      throw new Error(`Theme "${themeName}" not found`)
    }

    return JSON.stringify({
      name: themeName,
      config: themeToExport,
      colors: Array.from(customThemeColors.entries()),
    }, null, 2)
  }, [customThemes, customThemeColors])

  const importTheme = useCallback((themeJson: string) => {
    try {
      const parsed = JSON.parse(themeJson)
      if (parsed.name && parsed.config) {
        addCustomTheme(parsed.name, parsed.config)
        if (parsed.colors) {
          parsed.colors.forEach(([key, color]: [string, string]) => {
            setCustomColor(key, color)
          })
        }
      }
    } catch (error) {
      console.error('Error importing theme:', error)
      throw new Error('Invalid theme format')
    }
  }, [addCustomTheme, setCustomColor])

  return {
    // State
    theme,
    customThemes,
    activeTheme,
    customThemeColors,
    themePresets,
    // Actions
    setTheme,
    setActiveTheme,
    addCustomTheme,
    removeCustomTheme,
    updateCustomTheme,
    setCustomColor,
    addThemePreset,
    removeThemePreset,
    applyTheme,
    exportTheme,
    importTheme,
  }
}




