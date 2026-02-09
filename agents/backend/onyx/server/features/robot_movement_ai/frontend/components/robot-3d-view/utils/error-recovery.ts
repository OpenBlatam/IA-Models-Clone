/**
 * Error recovery system
 * @module robot-3d-view/utils/error-recovery
 */

/**
 * Error recovery strategy
 */
export type ErrorRecoveryStrategy =
  | 'retry'
  | 'fallback'
  | 'ignore'
  | 'notify'
  | 'rollback';

/**
 * Error recovery options
 */
export interface ErrorRecoveryOptions {
  strategy: ErrorRecoveryStrategy;
  maxRetries?: number;
  retryDelay?: number;
  fallbackValue?: unknown;
  onRecovery?: (error: Error) => void;
}

/**
 * Error Recovery Manager class
 */
export class ErrorRecoveryManager {
  private errorHistory: Error[] = [];
  private maxHistorySize = 100;
  private recoveryStrategies: Map<string, ErrorRecoveryOptions> = new Map();

  /**
   * Registers a recovery strategy for an error type
   */
  registerStrategy(
    errorType: string,
    options: ErrorRecoveryOptions
  ): void {
    this.recoveryStrategies.set(errorType, options);
  }

  /**
   * Handles an error with recovery
   */
  async handleError<T>(
    error: Error,
    operation: () => Promise<T>,
    options?: ErrorRecoveryOptions
  ): Promise<T | null> {
    // Add to history
    this.addToHistory(error);

    // Get strategy
    const strategy = options || this.recoveryStrategies.get(error.constructor.name);

    if (!strategy) {
      // Default: notify
      this.notifyError(error);
      return null;
    }

    switch (strategy.strategy) {
      case 'retry':
        return this.retry(operation, strategy);

      case 'fallback':
        return strategy.fallbackValue as T;

      case 'ignore':
        return null;

      case 'notify':
        this.notifyError(error);
        return null;

      case 'rollback':
        return this.rollback(error) as T;

      default:
        return null;
    }
  }

  /**
   * Retries an operation
   */
  private async retry<T>(
    operation: () => Promise<T>,
    options: ErrorRecoveryOptions
  ): Promise<T | null> {
    const maxRetries = options.maxRetries ?? 3;
    const retryDelay = options.retryDelay ?? 1000;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        if (attempt < maxRetries - 1) {
          await this.delay(retryDelay * (attempt + 1)); // Exponential backoff
        } else {
          this.notifyError(error as Error);
          return null;
        }
      }
    }

    return null;
  }

  /**
   * Rolls back to previous state
   */
  private rollback(error: Error): unknown {
    // Implementation would depend on state management
    this.notifyError(error);
    return null;
  }

  /**
   * Notifies about error
   */
  private notifyError(error: Error): void {
    console.error('Error recovery:', error);
    // Could integrate with notification system
  }

  /**
   * Adds error to history
   */
  private addToHistory(error: Error): void {
    this.errorHistory.push(error);
    if (this.errorHistory.length > this.maxHistorySize) {
      this.errorHistory.shift();
    }
  }

  /**
   * Gets error history
   */
  getErrorHistory(): Error[] {
    return [...this.errorHistory];
  }

  /**
   * Clears error history
   */
  clearErrorHistory(): void {
    this.errorHistory = [];
  }

  /**
   * Gets error statistics
   */
  getErrorStats(): {
    total: number;
    byType: Record<string, number>;
    recent: Error[];
  } {
    const byType: Record<string, number> = {};
    this.errorHistory.forEach((error) => {
      const type = error.constructor.name;
      byType[type] = (byType[type] || 0) + 1;
    });

    return {
      total: this.errorHistory.length,
      byType,
      recent: this.errorHistory.slice(-10),
    };
  }

  /**
   * Delay utility
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

/**
 * Global error recovery manager instance
 */
export const errorRecoveryManager = new ErrorRecoveryManager();

// Register default strategies
errorRecoveryManager.registerStrategy('NetworkError', {
  strategy: 'retry',
  maxRetries: 3,
  retryDelay: 1000,
});

errorRecoveryManager.registerStrategy('ValidationError', {
  strategy: 'notify',
});

errorRecoveryManager.registerStrategy('TypeError', {
  strategy: 'fallback',
  fallbackValue: null,
});



