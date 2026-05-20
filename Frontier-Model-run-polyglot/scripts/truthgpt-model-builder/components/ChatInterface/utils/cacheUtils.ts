/**
 * Advanced caching utilities
 */

export interface CacheOptions {
  ttl?: number // Time to live in milliseconds
  maxSize?: number // Maximum number of items
  strategy?: 'lru' | 'fifo' | 'lfu' // Cache eviction strategy
}

export interface CacheEntry<T> {
  key: string
  value: T
  timestamp: number
  accessCount: number
  lastAccessed: number
}

/**
 * LRU Cache implementation
 */
export class LRUCache<T> {
  private cache: Map<string, CacheEntry<T>> = new Map()
  private maxSize: number
  private ttl?: number

  constructor(options: CacheOptions = {}) {
    this.maxSize = options.maxSize || 100
    this.ttl = options.ttl
  }

  get(key: string): T | undefined {
    const entry = this.cache.get(key)
    
    if (!entry) {
      return undefined
    }

    // Check TTL
    if (this.ttl && Date.now() - entry.timestamp > this.ttl) {
      this.cache.delete(key)
      return undefined
    }

    // Update access info
    entry.lastAccessed = Date.now()
    entry.accessCount++

    // Move to end (most recently used)
    this.cache.delete(key)
    this.cache.set(key, entry)

    return entry.value
  }

  set(key: string, value: T): void {
    // Remove if exists
    if (this.cache.has(key)) {
      this.cache.delete(key)
    }

    // Check size limit
    if (this.cache.size >= this.maxSize) {
      // Remove least recently used (first item)
      const firstKey = this.cache.keys().next().value
      if (firstKey) {
        this.cache.delete(firstKey)
      }
    }

    // Add new entry
    this.cache.set(key, {
      key,
      value,
      timestamp: Date.now(),
      accessCount: 1,
      lastAccessed: Date.now(),
    })
  }

  delete(key: string): boolean {
    return this.cache.delete(key)
  }

  clear(): void {
    this.cache.clear()
  }

  has(key: string): boolean {
    const entry = this.cache.get(key)
    if (!entry) return false

    // Check TTL
    if (this.ttl && Date.now() - entry.timestamp > this.ttl) {
      this.cache.delete(key)
      return false
    }

    return true
  }

  size(): number {
    return this.cache.size
  }

  keys(): string[] {
    return Array.from(this.cache.keys())
  }

  values(): T[] {
    return Array.from(this.cache.values()).map(entry => entry.value)
  }

  entries(): Array<[string, T]> {
    return Array.from(this.cache.entries()).map(([key, entry]) => [key, entry.value])
  }

  getStats(): {
    size: number
    hitRate?: number
    averageAccessCount: number
  } {
    const entries = Array.from(this.cache.values())
    const totalAccessCount = entries.reduce((sum, entry) => sum + entry.accessCount, 0)
    
    return {
      size: this.cache.size,
      averageAccessCount: entries.length > 0 ? totalAccessCount / entries.length : 0,
    }
  }
}

/**
 * Create cache instance
 */
export function createCache<T>(options: CacheOptions = {}): LRUCache<T> {
  return new LRUCache<T>(options)
}

/**
 * Cache decorator for functions
 */
export function cached<T extends (...args: any[]) => any>(
  fn: T,
  options: CacheOptions & { keyFn?: (...args: Parameters<T>) => string } = {}
): T {
  const cache = createCache<any>(options)
  const keyFn = options.keyFn || ((...args: Parameters<T>) => JSON.stringify(args))

  return ((...args: Parameters<T>) => {
    const key = keyFn(...args)
    const cached = cache.get(key)
    
    if (cached !== undefined) {
      return cached
    }

    const result = fn(...args)
    cache.set(key, result)
    return result
  }) as T
}

/**
 * Multi-level cache
 */
export class MultiLevelCache<T> {
  private l1: LRUCache<T> // Fast, small cache
  private l2: LRUCache<T> // Slower, larger cache

  constructor(l1Options: CacheOptions = {}, l2Options: CacheOptions = {}) {
    this.l1 = new LRUCache<T>({ maxSize: 50, ...l1Options })
    this.l2 = new LRUCache<T>({ maxSize: 500, ...l2Options })
  }

  get(key: string): T | undefined {
    // Try L1 first
    const l1Value = this.l1.get(key)
    if (l1Value !== undefined) {
      return l1Value
    }

    // Try L2
    const l2Value = this.l2.get(key)
    if (l2Value !== undefined) {
      // Promote to L1
      this.l1.set(key, l2Value)
      return l2Value
    }

    return undefined
  }

  set(key: string, value: T): void {
    this.l1.set(key, value)
    this.l2.set(key, value)
  }

  clear(): void {
    this.l1.clear()
    this.l2.clear()
  }

  getStats(): {
    l1: ReturnType<LRUCache<T>['getStats']>
    l2: ReturnType<LRUCache<T>['getStats']>
  } {
    return {
      l1: this.l1.getStats(),
      l2: this.l2.getStats(),
    }
  }
}




