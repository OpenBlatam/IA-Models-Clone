/**
 * Smart Cache
 * Sistema de caché inteligente
 */

export interface CacheEntry<T> {
  key: string
  value: T
  timestamp: number
  expiresAt?: number
  accessCount: number
  lastAccessed: number
  size: number
}

export interface CacheOptions {
  maxSize?: number
  maxAge?: number
  strategy?: 'lru' | 'lfu' | 'fifo'
}

export class SmartCache {
  private cache: Map<string, CacheEntry<any>> = new Map()
  private maxSize: number = 100
  private maxAge: number = 3600000 // 1 hora
  private strategy: 'lru' | 'lfu' | 'fifo' = 'lru'
  private accessOrder: string[] = []
  private accessCounts: Map<string, number> = new Map()

  constructor(options: CacheOptions = {}) {
    this.maxSize = options.maxSize || 100
    this.maxAge = options.maxAge || 3600000
    this.strategy = options.strategy || 'lru'
  }

  /**
   * Obtener valor del caché
   */
  get<T>(key: string): T | null {
    const entry = this.cache.get(key)

    if (!entry) {
      return null
    }

    // Verificar expiración
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      this.delete(key)
      return null
    }

    // Actualizar acceso
    entry.accessCount++
    entry.lastAccessed = Date.now()
    this.updateAccessOrder(key)

    return entry.value as T
  }

  /**
   * Establecer valor en caché
   */
  set<T>(key: string, value: T, ttl?: number): void {
    const now = Date.now()
    const expiresAt = ttl ? now + ttl : undefined

    // Verificar tamaño
    if (this.cache.size >= this.maxSize) {
      this.evictEntry()
    }

    const entry: CacheEntry<T> = {
      key,
      value,
      timestamp: now,
      expiresAt,
      accessCount: 0,
      lastAccessed: now,
      size: this.estimateSize(value),
    }

    this.cache.set(key, entry)
    this.updateAccessOrder(key)
  }

  /**
   * Eliminar entrada
   */
  delete(key: string): boolean {
    const deleted = this.cache.delete(key)
    if (deleted) {
      const index = this.accessOrder.indexOf(key)
      if (index > -1) {
        this.accessOrder.splice(index, 1)
      }
      this.accessCounts.delete(key)
    }
    return deleted
  }

  /**
   * Limpiar caché
   */
  clear(): void {
    this.cache.clear()
    this.accessOrder = []
    this.accessCounts.clear()
  }

  /**
   * Verificar si existe
   */
  has(key: string): boolean {
    const entry = this.cache.get(key)
    if (!entry) return false

    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      this.delete(key)
      return false
    }

    return true
  }

  /**
   * Obtener todas las keys
   */
  keys(): string[] {
    return Array.from(this.cache.keys())
  }

  /**
   * Obtener estadísticas
   */
  getStats(): {
    size: number
    maxSize: number
    hitRate: number
    totalAccesses: number
    totalHits: number
    totalMisses: number
    averageAccessCount: number
  } {
    const entries = Array.from(this.cache.values())
    const totalAccesses = entries.reduce((sum, e) => sum + e.accessCount, 0)
    const totalHits = entries.length
    const totalMisses = 0 // Se debería trackear en producción
    const hitRate = totalAccesses > 0 ? totalHits / totalAccesses : 0
    const averageAccessCount =
      entries.length > 0 ? totalAccesses / entries.length : 0

    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hitRate,
      totalAccesses,
      totalHits,
      totalMisses,
      averageAccessCount,
    }
  }

  /**
   * Actualizar orden de acceso
   */
  private updateAccessOrder(key: string): void {
    const index = this.accessOrder.indexOf(key)
    if (index > -1) {
      this.accessOrder.splice(index, 1)
    }
    this.accessOrder.push(key)

    // Actualizar contador de acceso
    this.accessCounts.set(key, (this.accessCounts.get(key) || 0) + 1)
  }

  /**
   * Evictar entrada según estrategia
   */
  private evictEntry(): void {
    if (this.cache.size === 0) return

    let keyToEvict: string | null = null

    switch (this.strategy) {
      case 'lru':
        // Least Recently Used
        keyToEvict = this.accessOrder[0]
        break

      case 'lfu':
        // Least Frequently Used
        let minAccess = Infinity
        this.accessCounts.forEach((count, key) => {
          if (count < minAccess) {
            minAccess = count
            keyToEvict = key
          }
        })
        break

      case 'fifo':
        // First In First Out
        keyToEvict = this.accessOrder[0]
        break
    }

    if (keyToEvict) {
      this.delete(keyToEvict)
    }
  }

  /**
   * Estimar tamaño de valor
   */
  private estimateSize(value: any): number {
    try {
      return JSON.stringify(value).length
    } catch {
      return 1000 // Tamaño por defecto
    }
  }

  /**
   * Limpiar entradas expiradas
   */
  cleanExpired(): number {
    const now = Date.now()
    let cleaned = 0

    this.cache.forEach((entry, key) => {
      if (entry.expiresAt && now > entry.expiresAt) {
        this.delete(key)
        cleaned++
      }
    })

    return cleaned
  }

  /**
   * Obtener entradas ordenadas por acceso
   */
  getEntriesByAccess(): Array<{ key: string; accessCount: number }> {
    return Array.from(this.accessCounts.entries())
      .map(([key, count]) => ({ key, accessCount: count }))
      .sort((a, b) => b.accessCount - a.accessCount)
  }
}

// Singleton instances por estrategia
const cacheInstances: Map<string, SmartCache> = new Map()

export function getSmartCache(
  name: string = 'default',
  options?: CacheOptions
): SmartCache {
  if (!cacheInstances.has(name)) {
    cacheInstances.set(name, new SmartCache(options))
  }
  return cacheInstances.get(name)!
}

export default SmartCache










