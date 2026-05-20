/**
 * Semaphore utility.
 * Provides semaphore implementation for concurrency control.
 */

/**
 * Semaphore class.
 */
export class Semaphore {
  private permits: number;
  private waiting: Array<() => void> = [];

  constructor(initialPermits: number) {
    this.permits = initialPermits;
  }

  /**
   * Acquires a permit.
   */
  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }

    return new Promise<void>((resolve) => {
      this.waiting.push(resolve);
    });
  }

  /**
   * Releases a permit.
   */
  release(): void {
    if (this.waiting.length > 0) {
      const resolve = this.waiting.shift();
      resolve?.();
    } else {
      this.permits++;
    }
  }

  /**
   * Gets available permits.
   */
  availablePermits(): number {
    return this.permits;
  }

  /**
   * Gets waiting count.
   */
  getQueueLength(): number {
    return this.waiting.length;
  }

  /**
   * Executes function with semaphore.
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

/**
 * Creates a semaphore.
 */
export function createSemaphore(permits: number): Semaphore {
  return new Semaphore(permits);
}

