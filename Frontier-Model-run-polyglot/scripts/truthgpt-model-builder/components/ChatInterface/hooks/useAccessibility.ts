/**
 * Custom hook for accessibility features
 * Handles screen reader, high contrast, keyboard navigation, and other accessibility options
 */

import { useState, useCallback, useEffect } from 'react'

export interface AccessibilityState {
  accessibilityMode: boolean
  screenReader: boolean
  highContrast: boolean
  keyboardNavigation: boolean
  fontSize: 'small' | 'medium' | 'large'
  accessibilityFeatures: {
    screenReader: boolean
    highContrast: boolean
    largeText: boolean
    reducedMotion: boolean
  }
  readingSpeed: 'slow' | 'normal' | 'fast'
  zenMode: boolean
}

export interface AccessibilityActions {
  setAccessibilityMode: (enabled: boolean) => void
  setScreenReader: (enabled: boolean) => void
  setHighContrast: (enabled: boolean) => void
  setKeyboardNavigation: (enabled: boolean) => void
  setFontSize: (size: 'small' | 'medium' | 'large') => void
  updateAccessibilityFeature: (feature: keyof AccessibilityState['accessibilityFeatures'], enabled: boolean) => void
  setReadingSpeed: (speed: 'slow' | 'normal' | 'fast') => void
  setZenMode: (enabled: boolean) => void
  announceToScreenReader: (message: string) => void
}

const STORAGE_KEY = 'chat-accessibility-settings'

export function useAccessibility(): AccessibilityState & AccessibilityActions {
  const [accessibilityMode, setAccessibilityMode] = useState(false)
  const [screenReader, setScreenReader] = useState(false)
  const [highContrast, setHighContrast] = useState(false)
  const [keyboardNavigation, setKeyboardNavigation] = useState(true)
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large'>('medium')
  const [accessibilityFeatures, setAccessibilityFeatures] = useState({
    screenReader: false,
    highContrast: false,
    largeText: false,
    reducedMotion: false,
  })
  const [readingSpeed, setReadingSpeed] = useState<'slow' | 'normal' | 'fast'>('normal')
  const [zenMode, setZenMode] = useState(false)

  // Load from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        setAccessibilityMode(parsed.accessibilityMode || false)
        setScreenReader(parsed.screenReader || false)
        setHighContrast(parsed.highContrast || false)
        setKeyboardNavigation(parsed.keyboardNavigation !== false)
        setFontSize(parsed.fontSize || 'medium')
        setAccessibilityFeatures(parsed.accessibilityFeatures || {
          screenReader: false,
          highContrast: false,
          largeText: false,
          reducedMotion: false,
        })
        setReadingSpeed(parsed.readingSpeed || 'normal')
        setZenMode(parsed.zenMode || false)
      }
    } catch (error) {
      console.error('Error loading accessibility settings:', error)
    }
  }, [])

  // Save to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        accessibilityMode,
        screenReader,
        highContrast,
        keyboardNavigation,
        fontSize,
        accessibilityFeatures,
        readingSpeed,
        zenMode,
      }))
    } catch (error) {
      console.error('Error saving accessibility settings:', error)
    }
  }, [accessibilityMode, screenReader, highContrast, keyboardNavigation, fontSize, accessibilityFeatures, readingSpeed, zenMode])

  // Apply accessibility styles
  useEffect(() => {
    if (highContrast || accessibilityFeatures.highContrast) {
      document.documentElement.style.setProperty('--contrast', '1.2')
      document.documentElement.style.setProperty('filter', 'contrast(1.2)')
    } else {
      document.documentElement.style.removeProperty('--contrast')
      document.documentElement.style.removeProperty('filter')
    }

    if (accessibilityFeatures.largeText) {
      document.documentElement.style.setProperty('--font-size', '1.25rem')
      document.documentElement.style.setProperty('--line-height', '1.8')
    } else {
      document.documentElement.style.removeProperty('--font-size')
      document.documentElement.style.removeProperty('--line-height')
    }

    if (accessibilityFeatures.reducedMotion) {
      document.documentElement.style.setProperty('--animation-duration', '0.01ms')
      document.documentElement.style.setProperty('--transition-duration', '0.01ms')
    } else {
      document.documentElement.style.removeProperty('--animation-duration')
      document.documentElement.style.removeProperty('--transition-duration')
    }
  }, [highContrast, accessibilityFeatures])

  const updateAccessibilityFeature = useCallback((
    feature: keyof AccessibilityState['accessibilityFeatures'],
    enabled: boolean
  ) => {
    setAccessibilityFeatures(prev => ({
      ...prev,
      [feature]: enabled,
    }))
  }, [])

  const announceToScreenReader = useCallback((message: string) => {
    if (screenReader || accessibilityFeatures.screenReader) {
      const announcement = document.createElement('div')
      announcement.setAttribute('role', 'status')
      announcement.setAttribute('aria-live', 'polite')
      announcement.setAttribute('aria-atomic', 'true')
      announcement.className = 'sr-only'
      announcement.textContent = message
      document.body.appendChild(announcement)
      
      setTimeout(() => {
        document.body.removeChild(announcement)
      }, 1000)
    }
  }, [screenReader, accessibilityFeatures.screenReader])

  return {
    // State
    accessibilityMode,
    screenReader,
    highContrast,
    keyboardNavigation,
    fontSize,
    accessibilityFeatures,
    readingSpeed,
    zenMode,
    // Actions
    setAccessibilityMode,
    setScreenReader,
    setHighContrast,
    setKeyboardNavigation,
    setFontSize,
    updateAccessibilityFeature,
    setReadingSpeed,
    setZenMode,
    announceToScreenReader,
  }
}




