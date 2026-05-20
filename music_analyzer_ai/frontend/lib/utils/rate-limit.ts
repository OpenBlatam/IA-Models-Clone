/**
 * Rate limiting utility functions.
 * Provides helper functions for rate limiting operations.
 */

/**
 * Rate limiter class.
 */
export class RateLimiter {
  private queue: Array<() => void> = [];
  private executing = 0;
  private lastExecution = 0;

  constructor(
    private maxRequests: number,
    private windowMs: number
  ) {}

  /**
   * Executes function with rate limiting.
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.queue.push(async () => {
        try {
          const result = await fn();
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });

      this.processQueue();
    });
  }

  /**
   * Processes the queue.
   */
  private processQueue(): void {
    if (this.executing >= this.maxRequests || this.queue.length === 0) {
      return;
    }

    const now = Date.now();
    const timeSinceLastExecution = now - this.lastExecution;
    const timePerRequest = this.windowMs / this.maxRequests;

    if (timeSinceLastExecution < timePerRequest) {
      setTimeout(() => this.processQueue(), timePerRequest - timeSinceLastExecution);
      return;
    }

    const fn = this.queue.shift();
    if (fn) {
      this.executing++;
      this.lastExecution = Date.now();

      fn().finally(() => {
        this.executing--;
        this.processQueue();
      });
    }
  }

  /**
   * Clears the queue.
   */
  clear(): void {
    this.queue = [];
  }

  /**
   * Gets queue size.
   */
  getQueueSize(): number {
    return this.queue.length;
  }
}

/**
 * Creates a rate limiter.
 */
export function createRateLimiter(
  maxRequests: number,
  windowMs: number
): RateLimiter {
  return new RateLimiter(maxRequests, windowMs);
}

