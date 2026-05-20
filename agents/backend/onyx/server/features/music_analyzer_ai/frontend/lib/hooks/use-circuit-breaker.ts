/**
 * Custom hook for circuit breaker.
 * Provides reactive circuit breaker functionality.
 */

import { useRef, useCallback, useState } from 'react';
import {
  CircuitBreaker,
  createCircuitBreaker,
  CircuitBreakerOptions,
} from '../utils/circuit-breaker';

/**
 * Custom hook for circuit breaker.
 * Provides reactive circuit breaker functionality.
 *
 * @param options - Circuit breaker options
 * @returns Circuit breaker operations
 */
export function useCircuitBreaker(options: CircuitBreakerOptions = {}) {
  const breakerRef = useRef<CircuitBreaker>(createCircuitBreaker(options));
  const [state, setState] = useState(breakerRef.current.currentState);

  const execute = useCallback(
    async <T>(fn: () => Promise<T>): Promise<T> => {
      const result = await breakerRef.current.execute(fn);
      setState(breakerRef.current.currentState);
      return result;
    },
    []
  );

  const reset = useCallback(() => {
    breakerRef.current.reset();
    setState(breakerRef.current.currentState);
  }, []);

  return {
    state,
    execute,
    reset,
  };
}

