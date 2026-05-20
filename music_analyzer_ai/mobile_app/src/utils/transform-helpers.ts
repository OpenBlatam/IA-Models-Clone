/**
 * Transform utility functions
 * Data transformation operations
 */

/**
 * Transform array of objects
 */
export function transformArray<T, R>(
  array: T[],
  transformer: (item: T, index: number) => R
): R[] {
  return array.map(transformer);
}

/**
 * Transform object values
 */
export function transformObject<T extends Record<string, unknown>, R>(
  obj: T,
  transformer: (value: unknown, key: string) => R
): Record<string, R> {
  const result: Record<string, R> = {};
  for (const [key, value] of Object.entries(obj)) {
    result[key] = transformer(value, key);
  }
  return result;
}

/**
 * Transform object keys
 */
export function transformKeys<T extends Record<string, unknown>>(
  obj: T,
  transformer: (key: string) => string
): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    result[transformer(key)] = value;
  }
  return result;
}

/**
 * Transform and filter array
 */
export function transformAndFilter<T, R>(
  array: T[],
  transformer: (item: T, index: number) => R | null
): R[] {
  const result: R[] = [];
  for (let i = 0; i < array.length; i++) {
    const transformed = transformer(array[i], i);
    if (transformed !== null) {
      result.push(transformed);
    }
  }
  return result;
}

/**
 * Group array by key
 */
export function groupBy<T>(
  array: T[],
  keyFn: (item: T) => string
): Record<string, T[]> {
  const result: Record<string, T[]> = {};
  for (const item of array) {
    const key = keyFn(item);
    if (!result[key]) {
      result[key] = [];
    }
    result[key].push(item);
  }
  return result;
}

/**
 * Partition array
 */
export function partition<T>(
  array: T[],
  predicate: (item: T) => boolean
): [T[], T[]] {
  const truthy: T[] = [];
  const falsy: T[] = [];

  for (const item of array) {
    if (predicate(item)) {
      truthy.push(item);
    } else {
      falsy.push(item);
    }
  }

  return [truthy, falsy];
}

/**
 * Chunk array
 */
export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

/**
 * Flatten array
 */
export function flatten<T>(array: (T | T[])[]): T[] {
  const result: T[] = [];
  for (const item of array) {
    if (Array.isArray(item)) {
      result.push(...flatten(item));
    } else {
      result.push(item);
    }
  }
  return result;
}

/**
 * Unzip array of tuples
 */
export function unzip<T extends unknown[]>(
  array: T[]
): { [K in keyof T]: T[K][] } {
  if (array.length === 0) {
    return [] as unknown as { [K in keyof T]: T[K][] };
  }

  const length = array[0].length;
  const result = Array.from({ length }, () => []) as {
    [K in keyof T]: T[K][];
  };

  for (const item of array) {
    for (let i = 0; i < length; i++) {
      result[i].push(item[i]);
    }
  }

  return result;
}

