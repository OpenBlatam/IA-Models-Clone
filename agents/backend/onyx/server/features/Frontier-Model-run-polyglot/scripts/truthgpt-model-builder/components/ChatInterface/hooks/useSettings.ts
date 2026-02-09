/**
 * Custom hook for settings management
 * Handles all user preferences and settings
 */

import { useState, useCallback, useEffect } from 'react'

export interface SettingsState {
  // View settings
  viewMode: 'normal' | 'compact' | 'comfortable'
  fontSize: 'small' | 'medium' | 'large'
  compactMode: boolean
  showWordCount: boolean
  showSentiment: boolean
  showMessageTimestamps: boolean
  showTypingDots: boolean
  
  // Feature flags
  autoSave: boolean
  autoScroll: boolean
  autoFormat: boolean
  autoTranslate: boolean
  autoSummarize: boolean
  autoValidate: boolean
  autoComplete: boolean
  typingPrediction: boolean
  messageDeduplication: boolean
  messageCompression: boolean
  cacheEnabled: boolean
  smartSearch: boolean
  macroEnabled: boolean
  undoEnabled: boolean
  autoFilter: boolean
  customShortcutsEnabled: boolean
  
  // Display settings
  showCodeSyntax: boolean
  showReactions: boolean
  showTags: boolean
  showComments: boolean
  showAnnotations: boolean
  showPinnedMessages: boolean
  showBookmarks: boolean
  showStats: boolean
  showPerformance: boolean
  showDebug: boolean
  
  // Behavior settings
  readingSpeed: 'slow' | 'normal' | 'fast'
  messageFormatting: 'plain' | 'markdown' | 'html' | 'code'
  compressionLevel: 'low' | 'medium' | 'high'
  duplicateThreshold: number
  autoBackupInterval: number
  focusGoal: number
  
  // Advanced settings
  devMode: boolean
  realTimeStats: boolean
  advancedAnalytics: boolean
  sessionRecording: boolean
  cloudSync: boolean
  multiDeviceSync: boolean
  encryptionEnabled: boolean
  pluginSystem: boolean
}

const STORAGE_KEY = 'chat-interface-settings'

const DEFAULT_SETTINGS: SettingsState = {
  viewMode: 'normal',
  fontSize: 'medium',
  compactMode: false,
  showWordCount: false,
  showSentiment: false,
  showMessageTimestamps: true,
  showTypingDots: true,
  autoSave: true,
  autoScroll: true,
  autoFormat: true,
  autoTranslate: false,
  autoSummarize: false,
  autoValidate: false,
  autoComplete: true,
  typingPrediction: true,
  messageDeduplication: true,
  messageCompression: false,
  cacheEnabled: true,
  smartSearch: true,
  macroEnabled: true,
  undoEnabled: true,
  autoFilter: false,
  customShortcutsEnabled: true,
  showCodeSyntax: true,
  showReactions: true,
  showTags: true,
  showComments: false,
  showAnnotations: true,
  showPinnedMessages: true,
  showBookmarks: false,
  showStats: false,
  showPerformance: false,
  showDebug: false,
  readingSpeed: 'normal',
  messageFormatting: 'plain',
  compressionLevel: 'medium',
  duplicateThreshold: 0.8,
  autoBackupInterval: 300000,
  focusGoal: 25,
  devMode: false,
  realTimeStats: false,
  advancedAnalytics: false,
  sessionRecording: false,
  cloudSync: false,
  multiDeviceSync: false,
  encryptionEnabled: false,
  pluginSystem: false,
}

export function useSettings(): SettingsState & {
  updateSetting: <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => void
  updateSettings: (settings: Partial<SettingsState>) => void
  resetSettings: () => void
  exportSettings: () => string
  importSettings: (settingsJson: string) => void
} {
  const [settings, setSettings] = useState<SettingsState>(DEFAULT_SETTINGS)

  // Load settings from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        setSettings(prev => ({ ...DEFAULT_SETTINGS, ...parsed }))
      }
    } catch (error) {
      console.error('Error loading settings:', error)
    }
  }, [])

  // Save settings to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
    } catch (error) {
      console.error('Error saving settings:', error)
    }
  }, [settings])

  const updateSetting = useCallback(<K extends keyof SettingsState>(
    key: K,
    value: SettingsState[K]
  ) => {
    setSettings(prev => ({
      ...prev,
      [key]: value,
    }))
  }, [])

  const updateSettings = useCallback((newSettings: Partial<SettingsState>) => {
    setSettings(prev => ({
      ...prev,
      ...newSettings,
    }))
  }, [])

  const resetSettings = useCallback(() => {
    setSettings(DEFAULT_SETTINGS)
  }, [])

  const exportSettings = useCallback(() => {
    return JSON.stringify(settings, null, 2)
  }, [settings])

  const importSettings = useCallback((settingsJson: string) => {
    try {
      const parsed = JSON.parse(settingsJson)
      setSettings(prev => ({ ...DEFAULT_SETTINGS, ...parsed }))
    } catch (error) {
      console.error('Error importing settings:', error)
      throw new Error('Invalid settings format')
    }
  }, [])

  return {
    ...settings,
    updateSetting,
    updateSettings,
    resetSettings,
    exportSettings,
    importSettings,
  }
}




