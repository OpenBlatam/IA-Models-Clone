/**
 * Reducer utilities
 */

/**
 * Sum reducer
 */
export function sumReducer(acc: number, value: number): number {
  return acc + value;
}

/**
 * Product reducer
 */
export function productReducer(acc: number, value: number): number {
  return acc * value;
}

/**
 * Max reducer
 */
export function maxReducer<T>(acc: T, value: T): T {
  return value > acc ? value : acc;
}

/**
 * Min reducer
 */
export function minReducer<T>(acc: T, value: T): T {
  return value < acc ? value : acc;
}

/**
 * Count reducer
 */
export function countReducer(acc: number): number {
  return acc + 1;
}

/**
 * Concat reducer
 */
export function concatReducer<T>(acc: T[], value: T): T[] {
  return [...acc, value];
}

/**
 * Group reducer
 */
export function groupReducer<T, K extends string | number>(
  keyFn: (item: T) => K
): (acc: Record<K, T[]>, item: T) => Record<K, T[]> {
  return (acc: Record<K, T[]>, item: T) => {
    const key = keyFn(item);
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(item);
    return acc;
  };
}

/**
 * Create custom reducer
 */
export function createReducer<T, R>(
  initialValue: R,
  reducer: (acc: R, item: T) => R
): (items: T[]) => R {
  return (items: T[]) => items.reduce(reducer, initialValue);
}



