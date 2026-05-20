import { useEffect, useRef } from 'react';
import { deepClone } from '@/lib/utils/object';

export function useDeepCompareEffect(
  callback: React.EffectCallback,
  dependencies: React.DependencyList
) {
  const currentDependenciesRef = useRef<React.DependencyList>();

  if (!deepCompare(currentDependenciesRef.current, dependencies)) {
    currentDependenciesRef.current = dependencies;
  }

  useEffect(callback, currentDependenciesRef.current);
}

function deepCompare(a: any, b: any): boolean {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (typeof a !== 'object' || typeof b !== 'object') return false;

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (!keysB.includes(key)) return false;
    if (typeof a[key] === 'object' && typeof b[key] === 'object') {
      if (!deepCompare(a[key], b[key])) return false;
    } else if (a[key] !== b[key]) {
      return false;
    }
  }

  return true;
}



