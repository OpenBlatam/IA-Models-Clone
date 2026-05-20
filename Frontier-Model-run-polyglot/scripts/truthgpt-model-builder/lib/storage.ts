/**
 * Local Storage utilities for model persistence
 */

import { Model } from '@/store/modelStore'

const STORAGE_KEYS = {
  MODELS: 'truthgpt_models_history',
  PREFERENCES: 'truthgpt_preferences',
  RECENT_MODELS: 'truthgpt_recent_models',
}

export function saveModelToHistory(model: Model) {
  try {
    const history = getModelHistory()
    const existingIndex = history.findIndex(m => m.id === model.id)
    
    if (existingIndex >= 0) {
      history[existingIndex] = model
    } else {
      history.unshift(model)
    }
    
    // Keep only last 50 models
    const limitedHistory = history.slice(0, 50)
    localStorage.setItem(STORAGE_KEYS.MODELS, JSON.stringify(limitedHistory))
    return true
  } catch (error) {
    console.error('Error saving model to history:', error)
    return false
  }
}

export function getModelHistory(): Model[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.MODELS)
    if (!stored) return []
    
    const models = JSON.parse(stored)
    // Convert date strings back to Date objects
    return models.map((model: any) => ({
      ...model,
      createdAt: new Date(model.createdAt),
    }))
  } catch (error) {
    console.error('Error reading model history:', error)
    return []
  }
}

export function clearModelHistory() {
  try {
    localStorage.removeItem(STORAGE_KEYS.MODELS)
    return true
  } catch (error) {
    console.error('Error clearing model history:', error)
    return false
  }
}

export function deleteModelFromHistory(modelId: string) {
  try {
    const history = getModelHistory()
    const filtered = history.filter(m => m.id !== modelId)
    localStorage.setItem(STORAGE_KEYS.MODELS, JSON.stringify(filtered))
    return true
  } catch (error) {
    console.error('Error deleting model from history:', error)
    return false
  }
}

export function savePreferences(preferences: Record<string, any>) {
  try {
    const existing = getPreferences()
    const updated = { ...existing, ...preferences }
    localStorage.setItem(STORAGE_KEYS.PREFERENCES, JSON.stringify(updated))
    return true
  } catch (error) {
    console.error('Error saving preferences:', error)
    return false
  }
}

export function getPreferences(): Record<string, any> {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.PREFERENCES)
    return stored ? JSON.parse(stored) : {}
  } catch (error) {
    console.error('Error reading preferences:', error)
    return {}
  }
}


