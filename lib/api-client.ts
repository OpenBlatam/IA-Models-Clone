import { withRetry, withTimeout, AppError } from './error-handling';
import { openaiCircuitBreaker, stripeCircuitBreaker } from './circuit-breaker';
import { aiRateLimiter } from './rate-limiter';

export interface ApiClientConfig {
  baseURL?: string;
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
}

export class ApiClient {
  private config: Required<ApiClientConfig>;

  constructor(config: ApiClientConfig = {}) {
    this.config = {
      baseURL: config.baseURL || '',
      timeout: config.timeout || 30000,
      retries: config.retries || 3,
      headers: config.headers || {}
    };
  }

  async request<T>(
    endpoint: string,
    options: RequestInit & { 
      circuitBreaker?: string;
      rateLimitKey?: string;
      skipRetry?: boolean;
    } = {}
  ): Promise<T> {
    const { circuitBreaker, rateLimitKey, skipRetry, ...fetchOptions } = options;
    
    const url = this.config.baseURL + endpoint;
    
    const makeRequest = async (): Promise<T> => {
      if (rateLimitKey) {
        await aiRateLimiter.enforce(rateLimitKey);
      }

      const executeWithCircuitBreaker = circuitBreaker 
        ? this.getCircuitBreaker(circuitBreaker)?.execute.bind(this.getCircuitBreaker(circuitBreaker))
        : (fn: () => Promise<T>) => fn();

      return executeWithCircuitBreaker(async () => {
        const response = await withTimeout(
          fetch(url, {
            ...fetchOptions,
            headers: {
              'Content-Type': 'application/json',
              ...this.config.headers,
              ...fetchOptions.headers
            }
          }),
          this.config.timeout,
          `Request to ${endpoint} timed out after ${this.config.timeout}ms`
        );

        if (!response.ok) {
          const errorText = await response.text().catch(() => 'Unknown error');
          throw new AppError(
            `HTTP ${response.status}: ${errorText}`,
            response.status,
            true,
            { endpoint, status: response.status }
          );
        }

        const contentType = response.headers.get('content-type');
        if (contentType?.includes('application/json')) {
          return response.json();
        }
        
        return response.text() as T;
      });
    };

    if (skipRetry) {
      return makeRequest();
    }

    return withRetry(makeRequest, {
      maxAttempts: this.config.retries,
      baseDelay: 1000,
      exponentialBase: 2
    });
  }

  private getCircuitBreaker(name: string) {
    switch (name) {
      case 'openai':
        return openaiCircuitBreaker;
      case 'stripe':
        return stripeCircuitBreaker;
      default:
        return null;
    }
  }
}

export const apiClient = new ApiClient({
  baseURL: '/api',
  timeout: 30000,
  retries: 3
});
