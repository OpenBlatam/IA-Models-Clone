/**
 * Predicate utilities for filtering and checking
 */

/**
 * Create equality predicate
 */
export function equals<T>(value: T): (item: T) => boolean {
  return (item: T) => item === value;
}

/**
 * Create not equals predicate
 */
export function notEquals<T>(value: T): (item: T) => boolean {
  return (item: T) => item !== value;
}

/**
 * Create greater than predicate
 */
export function greaterThan(value: number): (item: number) => boolean {
  return (item: number) => item > value;
}

/**
 * Create less than predicate
 */
export function lessThan(value: number): (item: number) => boolean {
  return (item: number) => item < value;
}

/**
 * Create in range predicate
 */
export function inRange(min: number, max: number): (item: number) => boolean {
  return (item: number) => item >= min && item <= max;
}

/**
 * Create contains predicate
 */
export function contains<T>(value: T): (item: T[]) => boolean {
  return (item: T[]) => item.includes(value);
}

/**
 * Create matches predicate (regex)
 */
export function matches(pattern: RegExp | string): (item: string) => boolean {
  const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
  return (item: string) => regex.test(item);
}

/**
 * Create starts with predicate
 */
export function startsWith(prefix: string): (item: string) => boolean {
  return (item: string) => item.startsWith(prefix);
}

/**
 * Create ends with predicate
 */
export function endsWith(suffix: string): (item: string) => boolean {
  return (item: string) => item.endsWith(suffix);
}

/**
 * Create property equals predicate
 */
export function propertyEquals<T, K extends keyof T>(
  key: K,
  value: T[K]
): (item: T) => boolean {
  return (item: T) => item[key] === value;
}

/**
 * Create property in predicate
 */
export function propertyIn<T, K extends keyof T>(
  key: K,
  values: T[K][]
): (item: T) => boolean {
  return (item: T) => values.includes(item[key]);
}



