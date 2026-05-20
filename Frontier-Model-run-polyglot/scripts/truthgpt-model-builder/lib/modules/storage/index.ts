/**
 * Storage Module - Exports all storage-related functionality
 */

export { saveModelToHistory, getModelHistory, deleteModelFromHistory } from '../../storage'
export { saveDraft, getDraft, clearDraft, hasRecentDraft, setupAutoSave, type Draft } from '../../auto-save'
export { UnifiedCache, createCache, defaultCache, type CacheOptions, type CacheEntry } from './cache'

