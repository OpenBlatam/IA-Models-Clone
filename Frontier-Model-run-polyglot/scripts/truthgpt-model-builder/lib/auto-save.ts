/**
 * Auto-save and draft recovery system
 */

const DRAFT_KEY = 'truthgpt-model-builder-draft'
const DRAFT_TIMESTAMP_KEY = 'truthgpt-model-builder-draft-timestamp'
const AUTO_SAVE_INTERVAL = 2000 // 2 seconds

export interface Draft {
  input: string
  timestamp: Date
  modelId?: string
}

export function saveDraft(input: string, modelId?: string): void {
  if (typeof window === 'undefined') return

  try {
    const draft: Draft = {
      input,
      timestamp: new Date(),
      modelId,
    }
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
    localStorage.setItem(DRAFT_TIMESTAMP_KEY, Date.now().toString())
  } catch (error) {
    console.error('Error saving draft:', error)
  }
}

export function getDraft(): Draft | null {
  if (typeof window === 'undefined') return null

  try {
    const draftStr = localStorage.getItem(DRAFT_KEY)
    if (!draftStr) return null

    const draft = JSON.parse(draftStr)
    return {
      ...draft,
      timestamp: new Date(draft.timestamp),
    }
  } catch (error) {
    console.error('Error getting draft:', error)
    return null
  }
}

export function clearDraft(): void {
  if (typeof window === 'undefined') return

  try {
    localStorage.removeItem(DRAFT_KEY)
    localStorage.removeItem(DRAFT_TIMESTAMP_KEY)
  } catch (error) {
    console.error('Error clearing draft:', error)
  }
}

export function hasRecentDraft(maxAgeMinutes: number = 30): boolean {
  if (typeof window === 'undefined') return false

  try {
    const timestampStr = localStorage.getItem(DRAFT_TIMESTAMP_KEY)
    if (!timestampStr) return false

    const timestamp = parseInt(timestampStr, 10)
    const ageMinutes = (Date.now() - timestamp) / (1000 * 60)
    return ageMinutes < maxAgeMinutes
  } catch (error) {
    return false
  }
}

export function setupAutoSave(
  input: string,
  callback: (draft: Draft) => void,
  interval: number = AUTO_SAVE_INTERVAL
): () => void {
  if (typeof window === 'undefined') return () => {}

  let lastSaved = ''

  const intervalId = setInterval(() => {
    if (input !== lastSaved && input.trim().length > 0) {
      saveDraft(input)
      lastSaved = input
    }
  }, interval)

  return () => clearInterval(intervalId)
}


