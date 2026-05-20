/**
 * Custom hook for update-only effects.
 * Runs effect only on updates, not on initial mount.
 */

import { useEffect, useRef } from 'react';

/**
 * Custom hook for update-only effects.
 * Runs effect callback only when dependencies change, skipping initial mount.
 *
 * @param effect - Effect callback
 * @param deps - Dependency array
 */
export function useUpdateEffect(
  effect: React.EffectCallback,
  deps?: React.DependencyList
): void {
  const isFirstRender = useRef(true);

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }

    return effect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

