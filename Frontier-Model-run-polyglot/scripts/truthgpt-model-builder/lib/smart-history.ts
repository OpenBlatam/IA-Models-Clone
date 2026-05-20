/**
 * Smart History
 * Historial inteligente con búsqueda avanzada
 */

import { ProactiveBuildResult } from '../components/ProactiveModelBuilder'

export interface HistorySearchOptions {
  query?: string
  status?: 'all' | 'completed' | 'failed'
  dateRange?: {
    start: number
    end: number
  }
  minDuration?: number
  maxDuration?: number
  tags?: string[]
  categories?: string[]
  sortBy?: 'date' | 'duration' | 'name' | 'status'
  sortOrder?: 'asc' | 'desc'
  limit?: number
}

export interface HistoryGroup {
  date: string
  models: ProactiveBuildResult[]
  count: number
  successRate: number
}

export class SmartHistory {
  private models: ProactiveBuildResult[] = []
  private searchIndex: Map<string, Set<string>> = new Map()

  /**
   * Agregar modelo al historial
   */
  addModel(model: ProactiveBuildResult): void {
    this.models.push(model)
    this.updateSearchIndex(model)
  }

  /**
   * Actualizar índice de búsqueda
   */
  private updateSearchIndex(model: ProactiveBuildResult): void {
    const words = this.extractWords(model.description + ' ' + model.modelName)
    words.forEach(word => {
      if (!this.searchIndex.has(word)) {
        this.searchIndex.set(word, new Set())
      }
      this.searchIndex.get(word)!.add(model.modelId)
    })
  }

  /**
   * Extraer palabras de texto
   */
  private extractWords(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2)
  }

  /**
   * Buscar modelos
   */
  search(options: HistorySearchOptions = {}): ProactiveBuildResult[] {
    let results = [...this.models]

    // Filtrar por query
    if (options.query && options.query.trim()) {
      const queryWords = this.extractWords(options.query)
      const matchingIds = new Set<string>()

      queryWords.forEach(word => {
        const ids = this.searchIndex.get(word)
        if (ids) {
          ids.forEach(id => matchingIds.add(id))
        }
      })

      results = results.filter(model => matchingIds.has(model.modelId))
    }

    // Filtrar por status
    if (options.status && options.status !== 'all') {
      results = results.filter(model => model.status === options.status)
    }

    // Filtrar por rango de fechas
    if (options.dateRange) {
      results = results.filter(model => {
        const time = model.endTime || model.startTime || 0
        return time >= options.dateRange!.start && time <= options.dateRange!.end
      })
    }

    // Filtrar por duración
    if (options.minDuration !== undefined) {
      results = results.filter(model => model.duration && model.duration >= options.minDuration!)
    }
    if (options.maxDuration !== undefined) {
      results = results.filter(model => !model.duration || model.duration <= options.maxDuration!)
    }

    // Ordenar
    if (options.sortBy) {
      results.sort((a, b) => {
        let aValue: any
        let bValue: any

        switch (options.sortBy) {
          case 'date':
            aValue = a.endTime || a.startTime || 0
            bValue = b.endTime || b.startTime || 0
            break
          case 'duration':
            aValue = a.duration || 0
            bValue = b.duration || 0
            break
          case 'name':
            aValue = a.modelName.toLowerCase()
            bValue = b.modelName.toLowerCase()
            break
          case 'status':
            aValue = a.status
            bValue = b.status
            break
          default:
            return 0
        }

        const comparison = aValue < bValue ? -1 : aValue > bValue ? 1 : 0
        return options.sortOrder === 'desc' ? -comparison : comparison
      })
    }

    // Limitar resultados
    if (options.limit) {
      results = results.slice(0, options.limit)
    }

    return results
  }

  /**
   * Agrupar por fecha
   */
  groupByDate(models: ProactiveBuildResult[]): HistoryGroup[] {
    const groups = new Map<string, ProactiveBuildResult[]>()

    models.forEach(model => {
      const date = new Date(model.endTime || model.startTime || Date.now())
        .toISOString()
        .split('T')[0]
      
      if (!groups.has(date)) {
        groups.set(date, [])
      }
      groups.get(date)!.push(model)
    })

    return Array.from(groups.entries())
      .map(([date, models]) => ({
        date,
        models,
        count: models.length,
        successRate: models.filter(m => m.status === 'completed').length / models.length,
      }))
      .sort((a, b) => b.date.localeCompare(a.date))
  }

  /**
   * Obtener sugerencias de búsqueda
   */
  getSearchSuggestions(query: string, limit: number = 5): string[] {
    if (!query || query.length < 2) return []

    const queryLower = query.toLowerCase()
    const suggestions = new Set<string>()

    // Buscar en nombres de modelos
    this.models.forEach(model => {
      if (model.modelName.toLowerCase().includes(queryLower)) {
        suggestions.add(model.modelName)
      }
      if (model.description.toLowerCase().includes(queryLower)) {
        const words = this.extractWords(model.description)
        words.forEach(word => {
          if (word.includes(queryLower) && word.length > queryLower.length) {
            suggestions.add(word)
          }
        })
      }
    })

    return Array.from(suggestions).slice(0, limit)
  }

  /**
   * Obtener modelos recientes
   */
  getRecentModels(limit: number = 10): ProactiveBuildResult[] {
    return this.models
      .sort((a, b) => {
        const aTime = a.endTime || a.startTime || 0
        const bTime = b.endTime || b.startTime || 0
        return bTime - aTime
      })
      .slice(0, limit)
  }

  /**
   * Obtener modelos más rápidos
   */
  getFastestModels(limit: number = 10): ProactiveBuildResult[] {
    return this.models
      .filter(m => m.duration)
      .sort((a, b) => (a.duration || 0) - (b.duration || 0))
      .slice(0, limit)
  }

  /**
   * Obtener modelos más exitosos
   */
  getMostSuccessfulModels(limit: number = 10): ProactiveBuildResult[] {
    return this.models
      .filter(m => m.status === 'completed')
      .sort((a, b) => {
        const aTime = a.endTime || a.startTime || 0
        const bTime = b.endTime || b.startTime || 0
        return bTime - aTime
      })
      .slice(0, limit)
  }

  /**
   * Obtener estadísticas de búsqueda
   */
  getSearchStats(query: string): {
    totalMatches: number
    completed: number
    failed: number
    avgDuration: number
  } {
    const results = this.search({ query })
    const completed = results.filter(m => m.status === 'completed').length
    const failed = results.filter(m => m.status === 'failed').length
    const durations = results
      .filter(m => m.duration)
      .map(m => m.duration!)
    const avgDuration = durations.length > 0
      ? durations.reduce((sum, d) => sum + d, 0) / durations.length
      : 0

    return {
      totalMatches: results.length,
      completed,
      failed,
      avgDuration,
    }
  }

  /**
   * Limpiar historial
   */
  clear(): void {
    this.models = []
    this.searchIndex.clear()
  }

  /**
   * Obtener todos los modelos
   */
  getAllModels(): ProactiveBuildResult[] {
    return [...this.models]
  }
}

// Singleton instance
let smartHistoryInstance: SmartHistory | null = null

export function getSmartHistory(): SmartHistory {
  if (!smartHistoryInstance) {
    smartHistoryInstance = new SmartHistory()
  }
  return smartHistoryInstance
}

export default SmartHistory










