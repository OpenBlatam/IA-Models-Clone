/**
 * Custom hook for isomorphic layout effect.
 * Uses useLayoutEffect on client, useEffect on server.
 */

import { useEffect, useLayoutEffect } from 'react';

/**
 * Custom hook for isomorphic layout effect.
 * Automatically chooses useLayoutEffect or useEffect based on environment.
 *
 * @param effect - Effect callback
 * @param deps - Dependency array
 */
export function useIsomorphicLayoutEffect(
  effect: React.EffectCallback,
  deps?: React.DependencyList
): void {
  if (typeof window === 'undefined') {
    useEffect(effect, deps);
  } else {
    useLayoutEffect(effect, deps);
  }
}

