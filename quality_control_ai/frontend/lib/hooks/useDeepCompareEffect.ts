import { useEffect, useRef } from 'react';

const deepEqual = (a: unknown, b: unknown): boolean => {
  if (a === b) return true;

  if (a == null || b == null) return false;
  if (typeof a !== 'object' || typeof b !== 'object') return false;

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (!keysB.includes(key)) return false;
    if (!deepEqual((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key])) {
      return false;
    }
  }

  return true;
};

export const useDeepCompareEffect = (
  effect: React.EffectCallback,
  deps: React.DependencyList
): void => {
  const ref = useRef<React.DependencyList>();

  if (!ref.current || !deepEqual(ref.current, deps)) {
    ref.current = deps;
  }

  useEffect(effect, ref.current);
};

