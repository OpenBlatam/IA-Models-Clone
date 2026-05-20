/**
 * Advanced Caching System for TruthGPT
 * Provides TTL-based caching with LRU eviction
 */

interface CacheEntry<T> {
  value: T
  expiresAt: number
  createdAt: number
  accessCount: number
  lastAccessed: number
  size: number // in bytes
}

export class AdvancedCache<K, V> {
  private cache: Map<K, CacheEntry<V>> = new Map()
  private accessTimes: Map<K, number> = new Map()
  private maxSize: number
  private defaultTTL: number
  private hits = 0
  private misses = 0

  constructor(maxSize: number = 1000, defaultTTL: number = 3600000) {
    this.maxSize = maxSize
    this.defaultTTL = defaultTTL // 1 hour default
  }

  /**
   * Get value from cache
   */
  get(key: K): V | null {
    const entry = this.cache.get(key)
    
    if (!entry) {
      this.misses++
      return null
    }

    // Check if expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key)
      this.accessTimes.delete(key)
      this.misses++
      return null
    }

    // Update access tracking (LRU)
    entry.accessCount++
    entry.lastAccessed = Date.now()
    this.accessTimes.set(key, Date.now())
    this.hits++

    return entry.value
  }

  /**
   * Set value in cache
   */
  set(key: K, value: V, ttl?: number): void {
    const now = Date.now()
    const expiresAt = now + (ttl || this.defaultTTL)
    const size = this.estimateSize(value)

    // Evict LRU if cache is full
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this.evictLRU()
    }

    const entry: CacheEntry<V> = {
      value,
      expiresAt,
      createdAt: now,
      accessCount: 0,
      lastAccessed: now,
      size
    }

    this.cache.set(key, entry)
    this.accessTimes.set(key, now)
  }

  /**
   * Check if key exists and is valid
   */
  has(key: K): boolean {
    const entry = this.cache.get(key)
    if (!entry) return false
    
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key)
      this.accessTimes.delete(key)
      return false
    }

    return true
  }

  /**
   * Delete specific key
   */
  delete(key: K): boolean {
    const deleted = this.cache.delete(key)
    this.accessTimes.delete(key)
    return deleted
  }

  /**
   * Clear all cache
   */
  clear(): void {
    this.cache.clear()
    this.accessTimes.clear()
    this.hits = 0
    this.misses = 0
  }

  /**
   * Invalidate expired entries
   */
  cleanup(): number {
    const now = Date.now()
    let cleaned = 0

    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key)
        this.accessTimes.delete(key)
        cleaned++
      }
    }

    return cleaned
  }

  /**
   * Get cache statistics
   */
  getStats() {
    const totalRequests = this.hits + this.misses
    const hitRate = totalRequests > 0 ? (this.hits / totalRequests) * 100 : 0
    const totalSize = Array.from(this.cache.values())
      .reduce((sum, entry) => sum + entry.size, 0)

    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: Math.round(hitRate * 100) / 100,
      usagePercent: Math.round((this.cache.size / this.maxSize) * 100),
      totalSizeBytes: totalSize,
      totalSizeKB: Math.round((totalSize / 1024) * 100) / 100,
      totalSizeMB: Math.round((totalSize / (1024 * 1024)) * 100) / 100
    }
  }

  /**
   * Evict least recently used entry
   */
  private evictLRU(): void {
    if (this.accessTimes.size === 0) {
      // If no access times, remove oldest by creation time
      let oldestKey: K | null = null
      let oldestTime = Infinity

      for (const [key, entry] of this.cache.entries()) {
        if (entry.createdAt < oldestTime) {
          oldestTime = entry.createdAt
          oldestKey = key
        }
      }

      if (oldestKey !== null) {
        this.cache.delete(oldestKey)
      }
      return
    }

    // Find LRU key
    let lruKey: K | null = null
    let lruTime = Infinity

    for (const [key, accessTime] of this.accessTimes.entries()) {
      if (accessTime < lruTime) {
        lruTime = accessTime
        lruKey = key
      }
    }

    if (lruKey !== null) {
      this.cache.delete(lruKey)
      this.accessTimes.delete(lruKey)
    }
  }

  /**
   * Estimate size of value in bytes
   */
  private estimateSize(value: V): number {
    try {
      const str = JSON.stringify(value)
      return new Blob([str]).size
    } catch {
      // Fallback estimation
      return 1024 // 1KB default
    }
  }
}

// Global cache instance
export const messageCache = new AdvancedCache<string, any>(500, 1800000) // 30 minutes
export const modelCache = new AdvancedCache<string, any>(100, 7200000) // 2 hours


