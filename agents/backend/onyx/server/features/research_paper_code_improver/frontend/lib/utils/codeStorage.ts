/**
 * Code storage utilities for managing code history and related data
 */

import { storage, storageKeys } from './storage'

export interface CodeHistoryItem {
  id: string
  timestamp: string
  repo?: string
  filePath?: string
  originalCode: string
  improvedCode: string
  improvementsCount: number
  language: string
}

const MAX_HISTORY_ITEMS = 50

/**
 * Saves a code improvement result to history
 */
export function saveCodeToHistory(
  original: string,
  improved: string,
  suggestions: Array<{ type: string; description: string; line?: number; severity?: string }>,
  metadata: {
    repo?: string
    filePath?: string
    language: string
  }
): CodeHistoryItem {
  const history = getCodeHistory()
  const newItem: CodeHistoryItem = {
    id: Date.now().toString(),
    timestamp: new Date().toISOString(),
    repo: metadata.repo,
    filePath: metadata.filePath,
    originalCode: original,
    improvedCode: improved,
    improvementsCount: suggestions.length,
    language: metadata.language,
  }

  // Keep only the last MAX_HISTORY_ITEMS items
  const updatedHistory = [newItem, ...history].slice(0, MAX_HISTORY_ITEMS)
  storage.set(storageKeys.CODE_HISTORY, updatedHistory)

  return newItem
}

/**
 * Retrieves code improvement history
 */
export function getCodeHistory(): CodeHistoryItem[] {
  return storage.get<CodeHistoryItem[]>(storageKeys.CODE_HISTORY, []) || []
}

/**
 * Clears code improvement history
 */
export function clearCodeHistory(): void {
  storage.remove(storageKeys.CODE_HISTORY)
}

/**
 * Gets a specific history item by ID
 */
export function getHistoryItem(id: string): CodeHistoryItem | null {
  const history = getCodeHistory()
  return history.find((item) => item.id === id) || null
}

/**
 * Removes a specific history item
 */
export function removeHistoryItem(id: string): void {
  const history = getCodeHistory()
  const filtered = history.filter((item) => item.id !== id)
  storage.set(storageKeys.CODE_HISTORY, filtered)
}




