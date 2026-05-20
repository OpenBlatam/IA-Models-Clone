/**
 * LocalStorage utilities for persisting data
 */

const STORAGE_PREFIX = 'rpc-improver-'

export const storage = {
  set: <T>(key: string, value: T): void => {
    try {
      const serialized = JSON.stringify(value)
      localStorage.setItem(`${STORAGE_PREFIX}${key}`, serialized)
    } catch (error) {
      console.error('Error saving to localStorage:', error)
    }
  },

  get: <T>(key: string, defaultValue: T | null = null): T | null => {
    try {
      const item = localStorage.getItem(`${STORAGE_PREFIX}${key}`)
      if (item === null) {
        return defaultValue
      }
      return JSON.parse(item) as T
    } catch (error) {
      console.error('Error reading from localStorage:', error)
      return defaultValue
    }
  },

  remove: (key: string): void => {
    try {
      localStorage.removeItem(`${STORAGE_PREFIX}${key}`)
    } catch (error) {
      console.error('Error removing from localStorage:', error)
    }
  },

  clear: (): void => {
    try {
      const keys = Object.keys(localStorage)
      keys.forEach((key) => {
        if (key.startsWith(STORAGE_PREFIX)) {
          localStorage.removeItem(key)
        }
      })
    } catch (error) {
      console.error('Error clearing localStorage:', error)
    }
  },
}

export const storageKeys = {
  CODE_HISTORY: 'code-history',
  PREFERENCES: 'preferences',
  RECENT_PAPERS: 'recent-papers',
  FILTERS: 'filters',
} as const




