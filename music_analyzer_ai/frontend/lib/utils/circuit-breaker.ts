/**
 * Circuit breaker utility functions.
 * Provides helper functions for circuit breaker pattern.
 */

/**
 * Circuit breaker state.
 */
export type CircuitBreakerState = 'closed' | 'open' | 'half-open';

/**
 * Circuit breaker options.
 */
export interface CircuitBreakerOptions {
  failureThreshold?: number;
  resetTimeout?: number;
  monitoringPeriod?: number;
  onStateChange?: (state: CircuitBreakerState) => void;
}

/**
 * Circuit breaker class.
 */
export class CircuitBreaker {
  private state: CircuitBreakerState = 'closed';
  private failures = 0;
  private lastFailureTime = 0;
  private successCount = 0;

  constructor(
    private options: CircuitBreakerOptions = {}
  ) {
    const {
      failureThreshold = 5,
      resetTimeout = 60000,
      monitoringPeriod = 60000,
    } = options;

    this.options = {
      failureThreshold,
      resetTimeout,
      monitoringPeriod,
      ...options,
    };
  }

  /**
   * Gets current state.
   */
  get currentState(): CircuitBreakerState {
    return this.state;
  }

  /**
   * Executes function with circuit breaker.
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.lastFailureTime > this.options.resetTimeout!) {
        this.transitionTo('half-open');
      } else {
        throw new Error('Circuit breaker is open');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  /**
   * Handles success.
   */
  private onSuccess(): void {
    this.failures = 0;

    if (this.state === 'half-open') {
      this.successCount++;
      if (this.successCount >= 2) {
        this.transitionTo('closed');
        this.successCount = 0;
      }
    }
  }

  /**
   * Handles failure.
   */
  private onFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();

    if (this.state === 'half-open') {
      this.transitionTo('open');
    } else if (this.failures >= this.options.failureThreshold!) {
      this.transitionTo('open');
    }
  }

  /**
   * Transitions to new state.
   */
  private transitionTo(newState: CircuitBreakerState): void {
    if (this.state !== newState) {
      this.state = newState;
      if (this.options.onStateChange) {
        this.options.onStateChange(newState);
      }
    }
  }

  /**
   * Resets circuit breaker.
   */
  reset(): void {
    this.state = 'closed';
    this.failures = 0;
    this.successCount = 0;
    this.lastFailureTime = 0;
  }
}

/**
 * Creates a circuit breaker.
 */
export function createCircuitBreaker(
  options: CircuitBreakerOptions = {}
): CircuitBreaker {
  return new CircuitBreaker(options);
}

