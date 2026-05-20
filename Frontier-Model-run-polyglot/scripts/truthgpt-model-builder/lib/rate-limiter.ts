/**
 * Adaptive Rate Limiter for TruthGPT
 * Provides intelligent rate limiting with adaptive adjustment
 */

interface RateLimitConfig {
  rate: number // requests per second
  capacity: number // burst capacity
  minRate?: number
  maxRate?: number
}

interface RequestRecord {
  timestamp: number
  success: boolean
  duration: number
}

export class AdaptiveRateLimiter {
  private config: Required<RateLimitConfig>
  private tokens: number
  private lastRefill: number
  private requestHistory: RequestRecord[] = []
  private windowSize: number = 60000 // 1 minute
  private consecutiveFailures: number = 0
  private consecutiveSuccesses: number = 0

  constructor(config: RateLimitConfig) {
    this.config = {
      rate: config.rate,
      capacity: config.capacity,
      minRate: config.minRate || 1,
      maxRate: config.maxRate || 100
    }
    this.tokens = config.capacity
    this.lastRefill = Date.now()
  }

  /**
   * Check if request is allowed
   */
  canProceed(): boolean {
    this.refillTokens()
    return this.tokens >= 1
  }

  /**
   * Consume a token
   */
  consume(): boolean {
    if (!this.canProceed()) {
      return false
    }

    this.tokens -= 1
    return true
  }

  /**
   * Record request outcome for adaptive adjustment
   */
  recordRequest(success: boolean, duration: number): void {
    const now = Date.now()
    this.requestHistory.push({
      timestamp: now,
      success,
      duration
    })

    // Clean old history
    const cutoff = now - this.windowSize
    this.requestHistory = this.requestHistory.filter(r => r.timestamp > cutoff)

    // Update consecutive counters
    if (success) {
      this.consecutiveSuccesses++
      this.consecutiveFailures = 0
    } else {
      this.consecutiveFailures++
      this.consecutiveSuccesses = 0
    }

    // Adaptive adjustment
    this.adjustRate()
  }

  /**
   * Get current rate limit status
   */
  getStatus(): {
    allowed: boolean
    tokensAvailable: number
    tokensPerSecond: number
    waitTimeMs?: number
  } {
    this.refillTokens()
    const canProceed = this.tokens >= 1
    const waitTime = canProceed ? 0 : (1 - this.tokens) / this.config.rate * 1000

    return {
      allowed: canProceed,
      tokensAvailable: Math.max(0, this.tokens),
      tokensPerSecond: this.config.rate,
      waitTimeMs: waitTime > 0 ? Math.ceil(waitTime) : undefined
    }
  }

  /**
   * Get statistics
   */
  getStats(): {
    currentRate: number
    capacity: number
    tokensAvailable: number
    recentRequests: number
    successRate: number
    avgDuration: number
  } {
    this.refillTokens()
    
    const recent = this.requestHistory.filter(
      r => r.timestamp > Date.now() - this.windowSize
    )
    
    const successRate = recent.length > 0
      ? (recent.filter(r => r.success).length / recent.length) * 100
      : 100
    
    const avgDuration = recent.length > 0
      ? recent.reduce((sum, r) => sum + r.duration, 0) / recent.length
      : 0

    return {
      currentRate: this.config.rate,
      capacity: this.config.capacity,
      tokensAvailable: Math.max(0, this.tokens),
      recentRequests: recent.length,
      successRate: Math.round(successRate * 100) / 100,
      avgDuration: Math.round(avgDuration * 100) / 100
    }
  }

  /**
   * Refill tokens based on time elapsed
   */
  private refillTokens(): void {
    const now = Date.now()
    const elapsed = (now - this.lastRefill) / 1000 // seconds
    const tokensToAdd = elapsed * this.config.rate

    this.tokens = Math.min(
      this.config.capacity,
      this.tokens + tokensToAdd
    )

    this.lastRefill = now
  }

  /**
   * Adaptively adjust rate based on performance
   */
  private adjustRate(): void {
    // If too many failures, reduce rate
    if (this.consecutiveFailures >= 3) {
      this.config.rate = Math.max(
        this.config.minRate,
        this.config.rate * 0.8
      )
      this.consecutiveFailures = 0
      return
    }

    // If many successes, gradually increase rate
    if (this.consecutiveSuccesses >= 10) {
      this.config.rate = Math.min(
        this.config.maxRate,
        this.config.rate * 1.1
      )
      this.consecutiveSuccesses = 0
      return
    }

    // Adjust based on success rate
    const recent = this.requestHistory.filter(
      r => r.timestamp > Date.now() - 30000 // last 30 seconds
    )

    if (recent.length >= 10) {
      const successRate = recent.filter(r => r.success).length / recent.length
      
      if (successRate > 0.95) {
        // High success rate, can increase
        this.config.rate = Math.min(
          this.config.maxRate,
          this.config.rate * 1.05
        )
      } else if (successRate < 0.8) {
        // Low success rate, reduce
        this.config.rate = Math.max(
          this.config.minRate,
          this.config.rate * 0.9
        )
      }
    }
  }

  /**
   * Reset rate limiter
   */
  reset(): void {
    this.tokens = this.config.capacity
    this.lastRefill = Date.now()
    this.requestHistory = []
    this.consecutiveFailures = 0
    this.consecutiveSuccesses = 0
  }
}

// Global rate limiter instance
export const rateLimiter = new AdaptiveRateLimiter({
  rate: 10, // 10 requests per second
  capacity: 20, // burst capacity
  minRate: 1,
  maxRate: 50
})
