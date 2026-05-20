import { AppError, ExternalServiceError } from './error-handling';

export enum CircuitState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN',
  HALF_OPEN = 'HALF_OPEN'
}

export interface CircuitBreakerConfig {
  failureThreshold: number;
  recoveryTimeout: number;
  monitoringPeriod: number;
  expectedErrors?: (error: Error) => boolean;
}

export interface CircuitBreakerStats {
  state: CircuitState;
  failureCount: number;
  successCount: number;
  nextAttempt: number;
  totalRequests: number;
}

export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failureCount: number = 0;
  private successCount: number = 0;
  private nextAttempt: number = 0;
  private totalRequests: number = 0;

  constructor(
    private name: string,
    private config: CircuitBreakerConfig
  ) {}

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === CircuitState.OPEN) {
      if (Date.now() < this.nextAttempt) {
        throw new ExternalServiceError(
          `Circuit breaker ${this.name} is OPEN. Next attempt at ${new Date(this.nextAttempt).toISOString()}`
        );
      }
      this.state = CircuitState.HALF_OPEN;
    }

    this.totalRequests++;

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure(error as Error);
      throw error;
    }
  }

  private onSuccess(): void {
    this.successCount++;
    this.failureCount = 0;
    
    if (this.state === CircuitState.HALF_OPEN) {
      this.state = CircuitState.CLOSED;
      console.log(`Circuit breaker ${this.name} recovered to CLOSED state`);
    }
  }

  private onFailure(error: Error): void {
    if (this.config.expectedErrors && this.config.expectedErrors(error)) {
      return;
    }

    this.failureCount++;

    if (this.failureCount >= this.config.failureThreshold) {
      this.state = CircuitState.OPEN;
      this.nextAttempt = Date.now() + this.config.recoveryTimeout;
      
      console.error(
        `Circuit breaker ${this.name} opened due to ${this.failureCount} failures. ` +
        `Will retry at ${new Date(this.nextAttempt).toISOString()}`
      );
    }
  }

  getStats(): CircuitBreakerStats {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      nextAttempt: this.nextAttempt,
      totalRequests: this.totalRequests
    };
  }

  reset(): void {
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.nextAttempt = 0;
    console.log(`Circuit breaker ${this.name} manually reset`);
  }

  forceOpen(): void {
    this.state = CircuitState.OPEN;
    this.nextAttempt = Date.now() + this.config.recoveryTimeout;
    console.log(`Circuit breaker ${this.name} manually opened`);
  }
}

export const createCircuitBreaker = (
  name: string,
  config: Partial<CircuitBreakerConfig> = {}
): CircuitBreaker => {
  const defaultConfig: CircuitBreakerConfig = {
    failureThreshold: 5,
    recoveryTimeout: 60000, // 1 minute
    monitoringPeriod: 10000, // 10 seconds
    expectedErrors: (error: Error) => {
      return error instanceof AppError && error.statusCode < 500;
    }
  };

  return new CircuitBreaker(name, { ...defaultConfig, ...config });
};

export const openaiCircuitBreaker = createCircuitBreaker('openai', {
  failureThreshold: 3,
  recoveryTimeout: 30000,
});

export const databaseCircuitBreaker = createCircuitBreaker('database', {
  failureThreshold: 5,
  recoveryTimeout: 60000,
});

export const stripeCircuitBreaker = createCircuitBreaker('stripe', {
  failureThreshold: 3,
  recoveryTimeout: 45000,
});

export const s3CircuitBreaker = createCircuitBreaker('s3', {
  failureThreshold: 4,
  recoveryTimeout: 30000,
});
