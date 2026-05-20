/**
 * Favorites Manager
 * Sistema de marcadores/favoritos
 */

import { ProactiveBuildResult } from '../components/ProactiveModelBuilder'

export interface FavoriteModel extends ProactiveBuildResult {
  favoritedAt: number
  notes?: string
  tags?: string[]
}

export class FavoritesManager {
  private favorites: Map<string, FavoriteModel> = new Map()
  private storageKey = 'favorite-models'

  constructor() {
    this.loadFromStorage()
  }

  /**
   * Cargar desde almacenamiento
   */
  private loadFromStorage(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const stored = window.localStorage.getItem(this.storageKey)
        if (stored) {
          const favorites = JSON.parse(stored) as FavoriteModel[]
          favorites.forEach(fav => {
            this.favorites.set(fav.modelId, fav)
          })
        }
      }
    } catch (error) {
      console.error('Error loading favorites from storage:', error)
    }
  }

  /**
   * Guardar en almacenamiento
   */
  private saveToStorage(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const favorites = Array.from(this.favorites.values())
        window.localStorage.setItem(this.storageKey, JSON.stringify(favorites))
      }
    } catch (error) {
      console.error('Error saving favorites to storage:', error)
    }
  }

  /**
   * Agregar a favoritos
   */
  addFavorite(model: ProactiveBuildResult, notes?: string, tags?: string[]): FavoriteModel {
    const favorite: FavoriteModel = {
      ...model,
      favoritedAt: Date.now(),
      notes,
      tags: tags || [],
    }

    this.favorites.set(model.modelId, favorite)
    this.saveToStorage()

    return favorite
  }

  /**
   * Eliminar de favoritos
   */
  removeFavorite(modelId: string): boolean {
    const deleted = this.favorites.delete(modelId)
    if (deleted) {
      this.saveToStorage()
    }
    return deleted
  }

  /**
   * Verificar si es favorito
   */
  isFavorite(modelId: string): boolean {
    return this.favorites.has(modelId)
  }

  /**
   * Obtener favorito
   */
  getFavorite(modelId: string): FavoriteModel | undefined {
    return this.favorites.get(modelId)
  }

  /**
   * Obtener todos los favoritos
   */
  getAllFavorites(): FavoriteModel[] {
    return Array.from(this.favorites.values())
      .sort((a, b) => b.favoritedAt - a.favoritedAt)
  }

  /**
   * Buscar favoritos
   */
  searchFavorites(query: string): FavoriteModel[] {
    const queryLower = query.toLowerCase()
    return Array.from(this.favorites.values()).filter(fav =>
      fav.modelName.toLowerCase().includes(queryLower) ||
      fav.description.toLowerCase().includes(queryLower) ||
      fav.notes?.toLowerCase().includes(queryLower) ||
      fav.tags?.some(tag => tag.toLowerCase().includes(queryLower))
    )
  }

  /**
   * Actualizar favorito
   */
  updateFavorite(modelId: string, updates: Partial<FavoriteModel>): FavoriteModel | null {
    const favorite = this.favorites.get(modelId)
    if (!favorite) return null

    const updated: FavoriteModel = {
      ...favorite,
      ...updates,
    }

    this.favorites.set(modelId, updated)
    this.saveToStorage()

    return updated
  }

  /**
   * Obtener favoritos por tag
   */
  getFavoritesByTag(tag: string): FavoriteModel[] {
    return Array.from(this.favorites.values())
      .filter(fav => fav.tags?.includes(tag))
  }

  /**
   * Obtener todos los tags
   */
  getAllTags(): string[] {
    const tagsSet = new Set<string>()
    this.favorites.forEach(fav => {
      fav.tags?.forEach(tag => tagsSet.add(tag))
    })
    return Array.from(tagsSet).sort()
  }

  /**
   * Obtener estadísticas de favoritos
   */
  getStats(): {
    total: number
    byStatus: Record<string, number>
    byTag: Record<string, number>
    averageDuration: number
  } {
    const favorites = Array.from(this.favorites.values())
    const byStatus: Record<string, number> = {}
    const byTag: Record<string, number> = {}

    let totalDuration = 0
    let durationCount = 0

    favorites.forEach(fav => {
      byStatus[fav.status] = (byStatus[fav.status] || 0) + 1

      fav.tags?.forEach(tag => {
        byTag[tag] = (byTag[tag] || 0) + 1
      })

      if (fav.duration) {
        totalDuration += fav.duration
        durationCount++
      }
    })

    return {
      total: favorites.length,
      byStatus,
      byTag,
      averageDuration: durationCount > 0 ? totalDuration / durationCount : 0,
    }
  }

  /**
   * Limpiar favoritos
   */
  clear(): void {
    this.favorites.clear()
    this.saveToStorage()
  }
}

// Singleton instance
let favoritesManagerInstance: FavoritesManager | null = null

export function getFavoritesManager(): FavoritesManager {
  if (!favoritesManagerInstance) {
    favoritesManagerInstance = new FavoritesManager()
  }
  return favoritesManagerInstance
}

export default FavoritesManager










