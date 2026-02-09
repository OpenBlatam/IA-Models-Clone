/**
 * Configuration management
 */

import { DEFAULTS, STORAGE_KEYS } from '../utils/constants'

export interface ChatInterfaceConfig {
  // Storage
  storagePrefix: string
  enableLocalStorage: boolean
  enableSessionStorage: boolean
  
  // Performance
  enableVirtualScrolling: boolean
  virtualScrollOverscan: number
  debounceDelay: number
  throttleDelay: number
  
  // Features
  enableVoiceInput: boolean
  enableVoiceOutput: boolean
  enableExport: boolean
  enableImport: boolean
  enableCollaboration: boolean
  enableNotifications: boolean
  
  // UI
  defaultViewMode: 'normal' | 'compact' | 'comfortable'
  defaultTheme: 'dark' | 'light' | 'auto'
  defaultFontSize: 'small' | 'medium' | 'large'
  
  // Limits
  maxMessageLength: number
  maxSearchLength: number
  maxFileSize: number
  cacheTTL: number
  
  // Development
  enableDevMode: boolean
  enablePerformanceMonitoring: boolean
  enableErrorReporting: boolean
}

export const defaultConfig: ChatInterfaceConfig = {
  storagePrefix: 'chat-',
  enableLocalStorage: true,
  enableSessionStorage: false,
  enableVirtualScrolling: true,
  virtualScrollOverscan: 5,
  debounceDelay: DEFAULTS.DEBOUNCE_DELAY,
  throttleDelay: DEFAULTS.THROTTLE_DELAY,
  enableVoiceInput: false,
  enableVoiceOutput: false,
  enableExport: true,
  enableImport: true,
  enableCollaboration: false,
  enableNotifications: true,
  defaultViewMode: DEFAULTS.VIEW_MODE,
  defaultTheme: DEFAULTS.THEME,
  defaultFontSize: DEFAULTS.FONT_SIZE,
  maxMessageLength: DEFAULTS.MAX_MESSAGE_LENGTH,
  maxSearchLength: 200,
  maxFileSize: 10 * 1024 * 1024, // 10MB
  cacheTTL: DEFAULTS.CACHE_TTL,
  enableDevMode: false,
  enablePerformanceMonitoring: false,
  enableErrorReporting: false,
}

let currentConfig: ChatInterfaceConfig = { ...defaultConfig }

export function getConfig(): ChatInterfaceConfig {
  return { ...currentConfig }
}

export function updateConfig(updates: Partial<ChatInterfaceConfig>): void {
  currentConfig = {
    ...currentConfig,
    ...updates,
  }
  
  // Persist to localStorage if enabled
  if (currentConfig.enableLocalStorage) {
    try {
      localStorage.setItem(
        `${currentConfig.storagePrefix}config`,
        JSON.stringify(currentConfig)
      )
    } catch (error) {
      console.warn('Failed to save config to localStorage:', error)
    }
  }
}

export function loadConfig(): ChatInterfaceConfig {
  if (!currentConfig.enableLocalStorage) {
    return { ...defaultConfig }
  }

  try {
    const saved = localStorage.getItem(`${currentConfig.storagePrefix}config`)
    if (saved) {
      const parsed = JSON.parse(saved)
      currentConfig = { ...defaultConfig, ...parsed }
    }
  } catch (error) {
    console.warn('Failed to load config from localStorage:', error)
    currentConfig = { ...defaultConfig }
  }

  return { ...currentConfig }
}

export function resetConfig(): void {
  currentConfig = { ...defaultConfig }
  
  if (currentConfig.enableLocalStorage) {
    try {
      localStorage.removeItem(`${currentConfig.storagePrefix}config`)
    } catch (error) {
      console.warn('Failed to remove config from localStorage:', error)
    }
  }
}

// Initialize config on load
if (typeof window !== 'undefined') {
  loadConfig()
}




