/**
 * Custom hook for deep compare effect.
 * Runs effect only when dependencies change deeply.
 */

import { useEffect, useRef } from 'react';
import { isEqual } from '../utils/object';

/**
 * Custom hook for deep compare effect.
 * Runs effect only when dependencies change deeply.
 *
 * @param effect - Effect callback
 * @param deps - Dependency array
 */
export function useDeepCompareEffect(
  effect: React.EffectCallback,
  deps: React.DependencyList
): void {
  const ref = useRef<React.DependencyList>();

  if (!ref.current || !isEqual(ref.current, deps)) {
    ref.current = deps;
  }

  useEffect(effect, ref.current);
}

