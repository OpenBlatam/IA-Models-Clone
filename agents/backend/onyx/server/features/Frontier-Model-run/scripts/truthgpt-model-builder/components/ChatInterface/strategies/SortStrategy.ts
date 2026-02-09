/**
 * Strategy Pattern para algoritmos de ordenamiento
 * Permite cambiar el algoritmo de ordenamiento fácilmente
 */

export interface SortStrategy<T> {
  sort(items: T[], field: string, order: 'asc' | 'desc'): T[]
  getName(): string
}

/**
 * Estrategia de ordenamiento por timestamp
 */
export class TimestampSortStrategy<T extends { timestamp?: number }> implements SortStrategy<T> {
  sort(items: T[], field: string, order: 'asc' | 'desc'): T[] {
    return [...items].sort((a, b) => {
      const aValue = a.timestamp || 0
      const bValue = b.timestamp || 0
      return order === 'asc' ? aValue - bValue : bValue - aValue
    })
  }

  getName(): string {
    return 'timestamp'
  }
}

/**
 * Estrategia de ordenamiento por campo genérico
 */
export class FieldSortStrategy<T> implements SortStrategy<T> {
  sort(items: T[], field: string, order: 'asc' | 'desc'): T[] {
    return [...items].sort((a, b) => {
      const aValue = (a as any)[field]
      const bValue = (b as any)[field]

      if (aValue === undefined || aValue === null) return 1
      if (bValue === undefined || bValue === null) return -1

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return order === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue)
      }

      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return order === 'asc' ? aValue - bValue : bValue - aValue
      }

      return 0
    })
  }

  getName(): string {
    return 'field'
  }
}

/**
 * Estrategia de ordenamiento por prioridad
 */
export class PrioritySortStrategy<T extends { priority?: number }> implements SortStrategy<T> {
  sort(items: T[], field: string, order: 'asc' | 'desc'): T[] {
    return [...items].sort((a, b) => {
      const aValue = a.priority || 0
      const bValue = b.priority || 0
      return order === 'asc' ? aValue - bValue : bValue - aValue
    })
  }

  getName(): string {
    return 'priority'
  }
}

/**
 * Contexto que usa las estrategias
 */
export class SortContext<T> {
  private strategy: SortStrategy<T>

  constructor(strategy: SortStrategy<T>) {
    this.strategy = strategy
  }

  setStrategy(strategy: SortStrategy<T>): void {
    this.strategy = strategy
  }

  sort(items: T[], field: string, order: 'asc' | 'desc'): T[] {
    return this.strategy.sort(items, field, order)
  }

  getStrategyName(): string {
    return this.strategy.getName()
  }
}

/**
 * Factory para crear estrategias de ordenamiento
 */
export class SortStrategyFactory {
  static create<T>(type: 'timestamp' | 'field' | 'priority'): SortStrategy<T> {
    switch (type) {
      case 'timestamp':
        return new TimestampSortStrategy<T>() as SortStrategy<T>
      case 'priority':
        return new PrioritySortStrategy<T>() as SortStrategy<T>
      case 'field':
      default:
        return new FieldSortStrategy<T>()
    }
  }
}



