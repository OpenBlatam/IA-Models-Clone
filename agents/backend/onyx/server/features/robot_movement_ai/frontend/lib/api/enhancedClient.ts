import { apiClient } from './client';
import { offlineCache } from '../utils/offlineCache';
import { retry } from '../utils/retry';

// Enhanced API client with caching and retry
export class EnhancedAPIClient {
  private cacheEnabled = true;
  private retryEnabled = true;

  async getStatus() {
    return this.withCache('status', () => apiClient.getStatus());
  }

  async getMetrics() {
    return this.withCache('metrics', () => apiClient.getMetrics(), 30000); // 30s cache
  }

  async getHealth() {
    return this.withRetry(() => apiClient.getHealth());
  }

  async getResources() {
    return this.withCache('resources', () => apiClient.getResources(), 60000); // 1min cache
  }

  private async withCache<T>(
    key: string,
    fn: () => Promise<T>,
    ttl: number = 5000
  ): Promise<T> {
    if (this.cacheEnabled) {
      const cached = offlineCache.get<T>(key);
      if (cached !== null) {
        return cached;
      }
    }

    const data = await fn();
    if (this.cacheEnabled) {
      offlineCache.set(key, data, ttl);
    }
    return data;
  }

  private async withRetry<T>(fn: () => Promise<T>): Promise<T> {
    if (this.retryEnabled) {
      return retry(fn, {
        maxAttempts: 3,
        delay: 1000,
        backoff: 'exponential',
      });
    }
    return fn();
  }

  enableCache() {
    this.cacheEnabled = true;
  }

  disableCache() {
    this.cacheEnabled = false;
  }

  enableRetry() {
    this.retryEnabled = true;
  }

  disableRetry() {
    this.retryEnabled = false;
  }
}

export const enhancedAPIClient = new EnhancedAPIClient();


