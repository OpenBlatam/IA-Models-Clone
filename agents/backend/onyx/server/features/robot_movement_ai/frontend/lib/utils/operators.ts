/**
 * Functional programming operators
 */

/**
 * Identity function
 */
export function identity<T>(x: T): T {
  return x;
}

/**
 * Constant function
 */
export function constant<T>(x: T): () => T {
  return () => x;
}

/**
 * No-op function
 */
export function noop(): void {
  // Do nothing
}

/**
 * Always return true
 */
export function alwaysTrue(): boolean {
  return true;
}

/**
 * Always return false
 */
export function alwaysFalse(): boolean {
  return false;
}

/**
 * Negate a predicate function
 */
export function not<T>(
  predicate: (value: T) => boolean
): (value: T) => boolean {
  return (value: T) => !predicate(value);
}

/**
 * Logical AND for predicates
 */
export function and<T>(
  ...predicates: Array<(value: T) => boolean>
): (value: T) => boolean {
  return (value: T) => predicates.every((predicate) => predicate(value));
}

/**
 * Logical OR for predicates
 */
export function or<T>(
  ...predicates: Array<(value: T) => boolean>
): (value: T) => boolean {
  return (value: T) => predicates.some((predicate) => predicate(value));
}

/**
 * Get property from object
 */
export function prop<K extends string | number>(
  key: K
): <T extends Record<K, any>>(obj: T) => T[K] {
  return (obj) => obj[key];
}

/**
 * Get nested property from object
 */
export function path<T>(path: string[]): (obj: any) => T | undefined {
  return (obj: any) => {
    let current = obj;
    for (const key of path) {
      if (current === null || current === undefined) {
        return undefined;
      }
      current = current[key];
    }
    return current as T;
  };
}

/**
 * Compare two values
 */
export function compare<T>(a: T, b: T): number {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

/**
 * Compare by a key function
 */
export function compareBy<T>(
  keyFn: (value: T) => number | string
): (a: T, b: T) => number {
  return (a, b) => {
    const aVal = keyFn(a);
    const bVal = keyFn(b);
    return compare(aVal, bVal);
  };
}

/**
 * Create a range of numbers
 */
export function range(start: number, end: number, step: number = 1): number[] {
  const result: number[] = [];
  if (step > 0) {
    for (let i = start; i < end; i += step) {
      result.push(i);
    }
  } else if (step < 0) {
    for (let i = start; i > end; i += step) {
      result.push(i);
    }
  }
  return result;
}

/**
 * Repeat a value n times
 */
export function repeat<T>(value: T, times: number): T[] {
  return Array(times).fill(value);
}

/**
 * Create a function that returns the first argument
 */
export function first<T>(...args: T[]): T {
  return args[0];
}

/**
 * Create a function that returns the last argument
 */
export function last<T>(...args: T[]): T {
  return args[args.length - 1];
}



