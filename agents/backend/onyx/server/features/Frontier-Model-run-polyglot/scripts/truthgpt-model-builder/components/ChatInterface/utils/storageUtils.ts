/**
 * Utility functions for storage operations
 * Handles localStorage, sessionStorage, and IndexedDB
 */

export interface StorageOptions {
  ttl?: number // Time to live in milliseconds
  compress?: boolean
  encrypt?: boolean
}

/**
 * Safe localStorage get
 */
export function getFromStorage<T>(key: string, defaultValue: T): T {
  try {
    const item = localStorage.getItem(key)
    if (!item) return defaultValue

    const parsed = JSON.parse(item)
    
    // Check TTL if exists
    if (parsed.expires && Date.now() > parsed.expires) {
      localStorage.removeItem(key)
      return defaultValue
    }

    return parsed.value ?? defaultValue
  } catch (error) {
    console.error(`Error reading from storage (${key}):`, error)
    return defaultValue
  }
}

/**
 * Safe localStorage set
 */
export function setToStorage<T>(key: string, value: T, options: StorageOptions = {}): boolean {
  try {
    const data: any = { value }

    if (options.ttl) {
      data.expires = Date.now() + options.ttl
    }

    localStorage.setItem(key, JSON.stringify(data))
    return true
  } catch (error) {
    console.error(`Error writing to storage (${key}):`, error)
    
    // Try to free up space
    if (error instanceof DOMException && error.name === 'QuotaExceededError') {
      try {
        // Clear old items
        clearExpiredStorage()
        localStorage.setItem(key, JSON.stringify({ value }))
        return true
      } catch {
        return false
      }
    }
    
    return false
  }
}

/**
 * Remove from storage
 */
export function removeFromStorage(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch (error) {
    console.error(`Error removing from storage (${key}):`, error)
  }
}

/**
 * Clear expired items from storage
 */
export function clearExpiredStorage(): void {
  const now = Date.now()
  const keysToRemove: string[] = []

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (!key) continue

    try {
      const item = localStorage.getItem(key)
      if (!item) continue

      const parsed = JSON.parse(item)
      if (parsed.expires && now > parsed.expires) {
        keysToRemove.push(key)
      }
    } catch {
      // Ignore invalid items
    }
  }

  keysToRemove.forEach(key => localStorage.removeItem(key))
}

/**
 * Get storage size estimate
 */
export function getStorageSize(): { used: number, available: number, total: number } {
  let used = 0

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (key) {
      const value = localStorage.getItem(key) || ''
      used += key.length + value.length
    }
  }

  // Estimate available (usually 5-10MB)
  const total = 5 * 1024 * 1024 // 5MB estimate
  const available = total - used

  return {
    used,
    available,
    total,
  }
}

/**
 * Export storage data
 */
export function exportStorage(): string {
  const data: Record<string, any> = {}

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (key && key.startsWith('chat-') || key.startsWith('bulk-chat-')) {
      try {
        const value = localStorage.getItem(key)
        if (value) {
          data[key] = JSON.parse(value)
        }
      } catch {
        // Skip invalid items
      }
    }
  }

  return JSON.stringify(data, null, 2)
}

/**
 * Import storage data
 */
export function importStorage(dataJson: string): { success: number, failed: number } {
  let success = 0
  let failed = 0

  try {
    const data = JSON.parse(dataJson)

    for (const [key, value] of Object.entries(data)) {
      try {
        if (key.startsWith('chat-') || key.startsWith('bulk-chat-')) {
          localStorage.setItem(key, JSON.stringify(value))
          success++
        }
      } catch {
        failed++
      }
    }
  } catch (error) {
    console.error('Error importing storage:', error)
    failed++
  }

  return { success, failed }
}

/**
 * Clear all chat-related storage
 */
export function clearChatStorage(): void {
  const keysToRemove: string[] = []

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (key && (key.startsWith('chat-') || key.startsWith('bulk-chat-'))) {
      keysToRemove.push(key)
    }
  }

  keysToRemove.forEach(key => localStorage.removeItem(key))
}

/**
 * Batch operations for storage
 */
export function batchSetStorage(items: Array<{ key: string, value: any, options?: StorageOptions }>): {
  success: number
  failed: number
} {
  let success = 0
  let failed = 0

  for (const item of items) {
    if (setToStorage(item.key, item.value, item.options || {})) {
      success++
    } else {
      failed++
    }
  }

  return { success, failed }
}

export function batchGetStorage<T>(keys: string[], defaultValue: T): Map<string, T> {
  const results = new Map<string, T>()

  for (const key of keys) {
    results.set(key, getFromStorage(key, defaultValue))
  }

  return results
}




