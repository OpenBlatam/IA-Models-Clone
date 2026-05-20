import { RateLimitError } from './error-handling';

let Redis: any = null;
if (typeof window === 'undefined') {
  try {
    Redis = require('ioredis').Redis;
  } catch (error) {
    console.warn('Redis not available in this environment');
  }
}

export interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  keyGenerator?: (identifier: string) => string;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
}

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetTime: number;
  totalHits: number;
}

export class RateLimiter {
  private redis: any | null = null;
  private memoryStore: Map<string, { count: number; resetTime: number }> = new Map();

  constructor(private config: RateLimitConfig) {
    console.log('Redis disabled, using memory store for rate limiting');
  }

  async checkLimit(identifier: string): Promise<RateLimitResult> {
    const key = this.config.keyGenerator 
      ? this.config.keyGenerator(identifier)
      : `rate_limit:${identifier}`;

    if (this.redis) {
      return this.checkRedisLimit(key);
    } else {
      return this.checkMemoryLimit(key);
    }
  }

  private async checkRedisLimit(key: string): Promise<RateLimitResult> {
    const now = Date.now();
    const windowStart = now - this.config.windowMs;

    const pipeline = this.redis!.pipeline();
    pipeline.zremrangebyscore(key, 0, windowStart);
    pipeline.zcard(key);
    pipeline.zadd(key, now, `${now}-${Math.random()}`);
    pipeline.expire(key, Math.ceil(this.config.windowMs / 1000));

    const results = await pipeline.exec();
    const count = results![1][1] as number;

    const allowed = count <= this.config.maxRequests;
    const remaining = Math.max(0, this.config.maxRequests - count);
    const resetTime = now + this.config.windowMs;

    return {
      allowed,
      remaining,
      resetTime,
      totalHits: count
    };
  }

  private checkMemoryLimit(key: string): RateLimitResult {
    const now = Date.now();
    const entry = this.memoryStore.get(key);

    if (!entry || now > entry.resetTime) {
      this.memoryStore.set(key, { count: 1, resetTime: now + this.config.windowMs });
      return {
        allowed: true,
        remaining: this.config.maxRequests - 1,
        resetTime: now + this.config.windowMs,
        totalHits: 1
      };
    }

    entry.count++;
    const allowed = entry.count <= this.config.maxRequests;
    const remaining = Math.max(0, this.config.maxRequests - entry.count);

    return {
      allowed,
      remaining,
      resetTime: entry.resetTime,
      totalHits: entry.count
    };
  }

  async enforce(identifier: string): Promise<void> {
    const result = await this.checkLimit(identifier);
    
    if (!result.allowed) {
      throw new RateLimitError(this.config.maxRequests, this.config.windowMs);
    }
  }

  cleanup(): void {
    const now = Date.now();
    this.memoryStore.forEach((entry, key) => {
      if (now > entry.resetTime) {
        this.memoryStore.delete(key);
      }
    });
  }
}

export const createRateLimiter = (config: RateLimitConfig): RateLimiter => {
  const limiter = new RateLimiter(config);
  
  if (typeof window === 'undefined' && !process.env.REDIS_URL) {
    setInterval(() => limiter.cleanup(), 60000);
  }
  
  return limiter;
};

export const apiRateLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  maxRequests: 100,
  keyGenerator: (ip: string) => `api:${ip}`
});

export const authRateLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  maxRequests: 5,
  keyGenerator: (ip: string) => `auth:${ip}`
});

export const aiRateLimiter = createRateLimiter({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 10,
  keyGenerator: (userId: string) => `ai:${userId}`
});
