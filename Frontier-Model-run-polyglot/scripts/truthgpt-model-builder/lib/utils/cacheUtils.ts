/**
 * Utilidades de Cache Avanzadas
 * ==============================
 * 
 * Funciones para manejo de cache con diferentes estrategias
 */

/**
 * Item de cache con metadata
 */
interface CacheItem<T> {
  value: T
  expires: number | null
  createdAt: number
  accessCount: number
  lastAccessed: number
}

/**
 * Cache con TTL y LRU
 */
export class LRUCache<T = any> {
  private cache: Map<string, CacheItem<T>> = new Map()
  private maxSize: number
  private defaultTTL: number | null

  constructor(maxSize: number = 100, defaultTTL: number | null = null) {
    this.maxSize = maxSize
    this.defaultTTL = defaultTTL
  }

  /**
   * Obtiene un valor del cache
   */
  get(key: string): T | null {
    const item = this.cache.get(key)

    if (!item) return null

    // Verificar expiración
    if (item.expires !== null && Date.now() > item.expires) {
      this.cache.delete(key)
      return null
    }

    // Actualizar metadata de acceso
    item.accessCount++
    item.lastAccessed = Date.now()

    // Mover al final (LRU)
    this.cache.delete(key)
    this.cache.set(key, item)

    return item.value
  }

  /**
   * Establece un valor en el cache
   */
  set(key: string, value: T, ttl: number | null = null): void {
    const expires = ttl !== null 
      ? Date.now() + ttl 
      : (this.defaultTTL !== null ? Date.now() + this.defaultTTL : null)

    // Si existe, actualizar
    if (this.cache.has(key)) {
      const item = this.cache.get(key)!
      item.value = value
      item.expires = expires
      item.lastAccessed = Date.now()
      this.cache.delete(key)
      this.cache.set(key, item)
      return
    }

    // Si está lleno, eliminar el menos usado
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value
      this.cache.delete(firstKey)
    }

    const item: CacheItem<T> = {
      value,
      expires,
      createdAt: Date.now(),
      accessCount: 1,
      lastAccessed: Date.now()
    }

    this.cache.set(key, item)
  }

  /**
   * Elimina un valor del cache
   */
  delete(key: string): boolean {
    return this.cache.delete(key)
  }

  /**
   * Limpia todo el cache
   */
  clear(): void {
    this.cache.clear()
  }

  /**
   * Limpia elementos expirados
   */
  cleanExpired(): number {
    let cleaned = 0
    const now = Date.now()

    for (const [key, item] of this.cache.entries()) {
      if (item.expires !== null && now > item.expires) {
        this.cache.delete(key)
        cleaned++
      }
    }

    return cleaned
  }

  /**
   * Obtiene el tamaño del cache
   */
  size(): number {
    return this.cache.size
  }

  /**
   * Verifica si existe una clave
   */
  has(key: string): boolean {
    const item = this.cache.get(key)
    if (!item) return false

    if (item.expires !== null && Date.now() > item.expires) {
      this.cache.delete(key)
      return false
    }

    return true
  }

  /**
   * Obtiene todas las claves
   */
  keys(): string[] {
    return Array.from(this.cache.keys())
  }

  /**
   * Obtiene estadísticas del cache
   */
  getStats(): {
    size: number
    maxSize: number
    hitRate: number
    oldestItem: string | null
    newestItem: string | null
  } {
    const items = Array.from(this.cache.entries())
    const totalAccess = items.reduce((sum, [, item]) => sum + item.accessCount, 0)
    const hitRate = items.length > 0 ? totalAccess / items.length : 0

    const sortedByCreated = [...items].sort((a, b) => 
      a[1].createdAt - b[1].createdAt
    )

    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hitRate,
      oldestItem: sortedByCreated[0]?.[0] || null,
      newestItem: sortedByCreated[sortedByCreated.length - 1]?.[0] || null
    }
  }
}

/**
 * Cache con función de generación automática
 */
export class MemoizedCache<T = any> {
  private cache: LRUCache<T>
  private generators: Map<string, () => T | Promise<T>> = new Map()

  constructor(maxSize: number = 100, defaultTTL: number | null = null) {
    this.cache = new LRUCache<T>(maxSize, defaultTTL)
  }

  /**
   * Registra una función generadora
   */
  register(key: string, generator: () => T | Promise<T>, ttl?: number): void {
    this.generators.set(key, generator)
  }

  /**
   * Obtiene un valor, generándolo si no existe
   */
  async get(key: string, ttl?: number): Promise<T | null> {
    let value = this.cache.get(key)

    if (value === null && this.generators.has(key)) {
      const generator = this.generators.get(key)!
      const generated = await generator()
      this.cache.set(key, generated, ttl)
      value = generated
    }

    return value
  }

  /**
   * Establece un valor
   */
  set(key: string, value: T, ttl?: number): void {
    this.cache.set(key, value, ttl)
  }

  /**
   * Invalida una clave
   */
  invalidate(key: string): void {
    this.cache.delete(key)
  }

  /**
   * Limpia todo
   */
  clear(): void {
    this.cache.clear()
    this.generators.clear()
  }
}

/**
 * Cache simple con TTL
 */
export function createSimpleCache<T = any>(defaultTTL: number = 60000) {
  const cache = new Map<string, { value: T; expires: number }>()

  return {
    get(key: string): T | null {
      const item = cache.get(key)
      if (!item) return null

      if (Date.now() > item.expires) {
        cache.delete(key)
        return null
      }

      return item.value
    },

    set(key: string, value: T, ttl: number = defaultTTL): void {
      cache.set(key, {
        value,
        expires: Date.now() + ttl
      })
    },

    delete(key: string): boolean {
      return cache.delete(key)
    },

    clear(): void {
      cache.clear()
    },

    cleanExpired(): number {
      let cleaned = 0
      const now = Date.now()

      for (const [key, item] of cache.entries()) {
        if (now > item.expires) {
          cache.delete(key)
          cleaned++
        }
      }

      return cleaned
    },

    size(): number {
      return cache.size
    }
  }
}






