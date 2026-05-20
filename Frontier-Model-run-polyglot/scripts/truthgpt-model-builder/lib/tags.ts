/**
 * Tag and category system for models
 */

export interface Tag {
  id: string
  name: string
  color: string
  category: 'domain' | 'architecture' | 'custom'
}

export interface ModelTags {
  modelId: string
  tags: string[]
  category?: string
}

const TAGS_KEY = 'truthgpt-model-tags'
const FAVORITES_KEY = 'truthgpt-model-favorites'

const DEFAULT_TAGS: Tag[] = [
  // Domain tags
  { id: 'nlp', name: 'NLP', color: '#3B82F6', category: 'domain' },
  { id: 'vision', name: 'Computer Vision', color: '#10B981', category: 'domain' },
  { id: 'audio', name: 'Audio', color: '#8B5CF6', category: 'domain' },
  { id: 'time-series', name: 'Time Series', color: '#F59E0B', category: 'domain' },
  { id: 'recommendation', name: 'Recommendation', color: '#EF4444', category: 'domain' },
  
  // Architecture tags
  { id: 'dense', name: 'Dense', color: '#6B7280', category: 'architecture' },
  { id: 'cnn', name: 'CNN', color: '#3B82F6', category: 'architecture' },
  { id: 'lstm', name: 'LSTM', color: '#10B981', category: 'architecture' },
  { id: 'transformer', name: 'Transformer', color: '#8B5CF6', category: 'architecture' },
  
  // Custom tags
  { id: 'production', name: 'Production', color: '#10B981', category: 'custom' },
  { id: 'experimental', name: 'Experimental', color: '#F59E0B', category: 'custom' },
  { id: 'research', name: 'Research', color: '#8B5CF6', category: 'custom' },
]

export function getDefaultTags(): Tag[] {
  return DEFAULT_TAGS
}

export function getAllTags(): Tag[] {
  if (typeof window === 'undefined') return DEFAULT_TAGS

  try {
    const customTagsStr = localStorage.getItem('truthgpt-custom-tags')
    if (!customTagsStr) return DEFAULT_TAGS

    const customTags = JSON.parse(customTagsStr)
    return [...DEFAULT_TAGS, ...customTags]
  } catch (error) {
    console.error('Error getting tags:', error)
    return DEFAULT_TAGS
  }
}

export function createTag(tag: Omit<Tag, 'id'>): Tag {
  const newTag: Tag = {
    ...tag,
    id: `tag-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  }

  if (typeof window !== 'undefined') {
    try {
      const customTagsStr = localStorage.getItem('truthgpt-custom-tags')
      const customTags = customTagsStr ? JSON.parse(customTagsStr) : []
      customTags.push(newTag)
      localStorage.setItem('truthgpt-custom-tags', JSON.stringify(customTags))
    } catch (error) {
      console.error('Error creating tag:', error)
    }
  }

  return newTag
}

export function getModelTags(modelId: string): string[] {
  if (typeof window === 'undefined') return []

  try {
    const tagsStr = localStorage.getItem(TAGS_KEY)
    if (!tagsStr) return []

    const tags: Record<string, ModelTags> = JSON.parse(tagsStr)
    return tags[modelId]?.tags || []
  } catch (error) {
    console.error('Error getting model tags:', error)
    return []
  }
}

export function setModelTags(modelId: string, tags: string[]): void {
  if (typeof window === 'undefined') return

  try {
    const tagsStr = localStorage.getItem(TAGS_KEY)
    const allTags: Record<string, ModelTags> = tagsStr ? JSON.parse(tagsStr) : {}

    allTags[modelId] = {
      modelId,
      tags,
    }

    localStorage.setItem(TAGS_KEY, JSON.stringify(allTags))
  } catch (error) {
    console.error('Error setting model tags:', error)
  }
}

export function addTagToModel(modelId: string, tagId: string): void {
  const currentTags = getModelTags(modelId)
  if (!currentTags.includes(tagId)) {
    setModelTags(modelId, [...currentTags, tagId])
  }
}

export function removeTagFromModel(modelId: string, tagId: string): void {
  const currentTags = getModelTags(modelId)
  setModelTags(modelId, currentTags.filter(id => id !== tagId))
}

export function isFavorite(modelId: string): boolean {
  if (typeof window === 'undefined') return false

  try {
    const favoritesStr = localStorage.getItem(FAVORITES_KEY)
    if (!favoritesStr) return false

    const favorites: string[] = JSON.parse(favoritesStr)
    return favorites.includes(modelId)
  } catch (error) {
    console.error('Error checking favorite:', error)
    return false
  }
}

export function toggleFavorite(modelId: string): boolean {
  if (typeof window === 'undefined') return false

  try {
    const favoritesStr = localStorage.getItem(FAVORITES_KEY)
    const favorites: string[] = favoritesStr ? JSON.parse(favoritesStr) : []

    const index = favorites.indexOf(modelId)
    if (index > -1) {
      favorites.splice(index, 1)
    } else {
      favorites.push(modelId)
    }

    localStorage.setItem(FAVORITES_KEY, JSON.stringify(favorites))
    return favorites.includes(modelId)
  } catch (error) {
    console.error('Error toggling favorite:', error)
    return false
  }
}

export function getFavorites(): string[] {
  if (typeof window === 'undefined') return []

  try {
    const favoritesStr = localStorage.getItem(FAVORITES_KEY)
    return favoritesStr ? JSON.parse(favoritesStr) : []
  } catch (error) {
    console.error('Error getting favorites:', error)
    return []
  }
}


