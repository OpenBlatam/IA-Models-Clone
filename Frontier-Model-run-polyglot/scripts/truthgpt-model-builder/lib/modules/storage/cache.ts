/**
 * Unified Cache Implementation
 * 
 * Consolidates all cache functionality into a single, well-structured module.
 * Supports multiple cache strategies: LRU, FIFO, and Time-based expiration.
 */

export interface CacheOptions {
  maxSize?: number
  ttl?: number
  strategy?: 'lru' | 'fifo' | 'time'
}

export interface CacheEntry<T> {
  key: string
  value: T
  timestamp: number
  accessCount: number
  lastAccessed: number
}

export class UnifiedCache<T = any> {
  private cache: Map<string, CacheEntry<T>>
  private maxSize: number
  private ttl: number
  private strategy: 'lru' | 'fifo' | 'time'

  constructor(options: CacheOptions = {}) {
    this.cache = new Map()
    this.maxSize = options.maxSize ?? 100
    this.ttl = options.ttl ?? 3600000
    this.strategy = options.strategy ?? 'lru'
  }

  get(key: string): T | undefined {
    const entry = this.cache.get(key)
    
    if (!entry) {
      return undefined
    }

    if (this.isExpired(entry)) {
      this.cache.delete(key)
      return undefined
    }

    entry.accessCount++
    entry.lastAccessed = Date.now()
    return entry.value
  }

  set(key: string, value: T): void {
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this.evict()
    }

    this.cache.set(key, {
      key,
      value,
      timestamp: Date.now(),
      accessCount: 1,
      lastAccessed: Date.now(),
    })
  }

  has(key: string): boolean {
    const entry = this.cache.get(key)
    if (!entry) return false
    
    if (this.isExpired(entry)) {
      this.cache.delete(key)
      return false
    }
    
    return true
  }

  delete(key: string): boolean {
    return this.cache.delete(key)
  }

  clear(): void {
    this.cache.clear()
  }

  size(): number {
    return this.cache.size
  }

  keys(): string[] {
    return Array.from(this.cache.keys())
  }

  private isExpired(entry: CacheEntry<T>): boolean {
    if (this.strategy === 'time') {
      return Date.now() - entry.timestamp > this.ttl
    }
    return false
  }

  private evict(): void {
    if (this.cache.size === 0) return

    let keyToEvict: string | null = null

    if (this.strategy === 'lru') {
      let oldestAccess = Infinity
      for (const [key, entry] of this.cache.entries()) {
        if (entry.lastAccessed < oldestAccess) {
          oldestAccess = entry.lastAccessed
          keyToEvict = key
        }
      }
    } else if (this.strategy === 'fifo') {
      let oldestTimestamp = Infinity
      for (const [key, entry] of this.cache.entries()) {
        if (entry.timestamp < oldestTimestamp) {
          oldestTimestamp = entry.timestamp
          keyToEvict = key
        }
      }
    } else if (this.strategy === 'time') {
      for (const [key, entry] of this.cache.entries()) {
        if (this.isExpired(entry)) {
          keyToEvict = key
          break
        }
      }
      if (!keyToEvict) {
        let oldestTimestamp = Infinity
        for (const [key, entry] of this.cache.entries()) {
          if (entry.timestamp < oldestTimestamp) {
            oldestTimestamp = entry.timestamp
            keyToEvict = key
          }
        }
      }
    }

    if (keyToEvict) {
      this.cache.delete(keyToEvict)
    }
  }
}

export const createCache = <T = any>(options?: CacheOptions) => {
  return new UnifiedCache<T>(options)
}

export const defaultCache = createCache()

