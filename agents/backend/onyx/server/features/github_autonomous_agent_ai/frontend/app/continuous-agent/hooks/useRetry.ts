/**
 * Custom hook for retry operations with exponential backoff
 * 
 * Provides retry functionality with React state management
 */

import { useState, useCallback } from "react";
import { retryWithBackoff, type RetryOptions } from "../utils/performance/retry";
import type { AgentError } from "../utils/errors/agent-errors";

/**
 * Return type for useRetry hook
 */
export interface UseRetryReturn<T> {
  /** Execute function with retry */
  readonly execute: () => Promise<T>;
  /** Whether operation is in progress */
  readonly isLoading: boolean;
  /** Current attempt number */
  readonly attempt: number;
  /** Last error encountered */
  readonly error: AgentError | null;
  /** Reset retry state */
  readonly reset: () => void;
}

/**
 * Custom hook for retry operations
 * 
 * @param fn - Async function to retry
 * @param options - Retry configuration options
 * @returns Retry state and execute function
 * 
 * @example
 * ```typescript
 * const { execute, isLoading, error } = useRetry(
 *   () => fetchAgent(id),
 *   { maxAttempts: 3, initialDelayMs: 1000 }
 * );
 * 
 * await execute();
 * ```
 */
export function useRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): UseRetryReturn<T> {
  const [isLoading, setIsLoading] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [error, setError] = useState<AgentError | null>(null);

  const execute = useCallback(async (): Promise<T> => {
    setIsLoading(true);
    setError(null);
    setAttempt(0);

    try {
      const result = await retryWithBackoff(fn, {
        ...options,
        onRetry: (currentAttempt, err) => {
          setAttempt(currentAttempt);
          setError(err as AgentError);
          options.onRetry?.(currentAttempt, err);
        },
      });

      setIsLoading(false);
      setAttempt(0);
      return result;
    } catch (err) {
      const agentError = err as AgentError;
      setError(agentError);
      setIsLoading(false);
      throw agentError;
    }
  }, [fn, options]);

  const reset = useCallback((): void => {
    setIsLoading(false);
    setAttempt(0);
    setError(null);
  }, []);

  return {
    execute,
    isLoading,
    attempt,
    error,
    reset,
  };
}




