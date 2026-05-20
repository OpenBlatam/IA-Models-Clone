import { useMemo, useRef } from 'react';

/**
 * Memoized value that only updates when the equality check passes
 * Useful for expensive computations or object comparisons
 */
export function useMemoizedValue<T>(
  value: T,
  equalityFn?: (prev: T, next: T) => boolean
): T {
  const ref = useRef<{ value: T; equalityFn?: (prev: T, next: T) => boolean }>({
    value,
    equalityFn,
  });

  return useMemo(() => {
    if (equalityFn) {
      if (!equalityFn(ref.current.value, value)) {
        ref.current.value = value;
      }
    } else if (ref.current.value !== value) {
      ref.current.value = value;
    }

    return ref.current.value;
  }, [value, equalityFn]);
}

/**
 * Deep equality check for objects
 */
function deepEqual<T>(a: T, b: T): boolean {
  if (a === b) {
    return true;
  }

  if (
    a === null ||
    b === null ||
    typeof a !== 'object' ||
    typeof b !== 'object'
  ) {
    return false;
  }

  const keysA = Object.keys(a as Record<string, unknown>);
  const keysB = Object.keys(b as Record<string, unknown>);

  if (keysA.length !== keysB.length) {
    return false;
  }

  for (const key of keysA) {
    if (!keysB.includes(key)) {
      return false;
    }

    if (
      !deepEqual(
        (a as Record<string, unknown>)[key],
        (b as Record<string, unknown>)[key]
      )
    ) {
      return false;
    }
  }

  return true;
}

/**
 * Memoized value with deep equality check
 */
export function useDeepMemoizedValue<T>(value: T): T {
  return useMemoizedValue(value, deepEqual);
}

