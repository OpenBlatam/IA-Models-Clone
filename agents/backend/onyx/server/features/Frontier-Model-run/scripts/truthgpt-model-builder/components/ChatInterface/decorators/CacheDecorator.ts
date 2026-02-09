/**
 * Decorator para agregar cache a operaciones
 */

interface CacheEntry<T> {
  value: T
  timestamp: number
  ttl?: number
}

export class CacheDecorator {
  private static cache = new Map<string, CacheEntry<any>>()

  /**
   * Decora una función con cache
   */
  static withCache<TArgs extends any[], TReturn>(
    operation: (...args: TArgs) => TReturn,
    cacheKey: (...args: TArgs) => string,
    ttl?: number
  ): (...args: TArgs) => TReturn {
    return (...args: TArgs): TReturn => {
      const key = cacheKey(...args)
      const cached = this.cache.get(key)

      // Verificar si el cache es válido
      if (cached) {
        if (ttl && Date.now() - cached.timestamp > ttl) {
          this.cache.delete(key)
        } else {
          return cached.value
        }
      }

      // Ejecutar operación y cachear resultado
      const result = operation(...args)
      this.cache.set(key, {
        value: result,
        timestamp: Date.now(),
        ttl
      })

      return result
    }
  }

  /**
   * Limpia el cache
   */
  static clearCache(): void {
    this.cache.clear()
  }

  /**
   * Limpia entradas expiradas
   */
  static clearExpired(): void {
    const now = Date.now()
    this.cache.forEach((entry, key) => {
      if (entry.ttl && now - entry.timestamp > entry.ttl) {
        this.cache.delete(key)
      }
    })
  }
}



