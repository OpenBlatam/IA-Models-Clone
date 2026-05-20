/**
 * Transform utility functions.
 * Provides helper functions for data transformation.
 */

/**
 * Maps array and transforms each item.
 * @param items - Array to map
 * @param transform - Transform function
 * @returns Transformed array
 */
export function map<T, U>(
  items: T[],
  transform: (item: T, index: number) => U
): U[] {
  return items.map(transform);
}

/**
 * Reduces array to a single value.
 * @param items - Array to reduce
 * @param reducer - Reducer function
 * @param initialValue - Initial value
 * @returns Reduced value
 */
export function reduce<T, U>(
  items: T[],
  reducer: (accumulator: U, item: T, index: number) => U,
  initialValue: U
): U {
  return items.reduce(reducer, initialValue);
}

/**
 * Groups array items by key.
 * @param items - Array to group
 * @param getKey - Function to get group key
 * @returns Grouped object
 */
export function groupBy<T, K extends string | number | symbol>(
  items: T[],
  getKey: (item: T) => K
): Record<K, T[]> {
  return items.reduce(
    (groups, item) => {
      const key = getKey(item);
      if (!groups[key]) {
        groups[key] = [];
      }
      groups[key].push(item);
      return groups;
    },
    {} as Record<K, T[]>
  );
}

/**
 * Partitions array into two arrays based on predicate.
 * @param items - Array to partition
 * @param predicate - Partition function
 * @returns Tuple of [true items, false items]
 */
export function partition<T>(
  items: T[],
  predicate: (item: T) => boolean
): [T[], T[]] {
  const truthy: T[] = [];
  const falsy: T[] = [];

  for (const item of items) {
    if (predicate(item)) {
      truthy.push(item);
    } else {
      falsy.push(item);
    }
  }

  return [truthy, falsy];
}

/**
 * Picks specified properties from object.
 * @param obj - Object to pick from
 * @param keys - Keys to pick
 * @returns New object with picked properties
 */
export function pick<T extends Record<string, any>, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) {
    if (key in obj) {
      result[key] = obj[key];
    }
  }
  return result;
}

/**
 * Omits specified properties from object.
 * @param obj - Object to omit from
 * @param keys - Keys to omit
 * @returns New object without omitted properties
 */
export function omit<T extends Record<string, any>, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj };
  for (const key of keys) {
    delete result[key];
  }
  return result;
}

/**
 * Transforms object keys.
 * @param obj - Object to transform
 * @param transformKey - Key transform function
 * @returns New object with transformed keys
 */
export function transformKeys<T extends Record<string, any>>(
  obj: T,
  transformKey: (key: string) => string
): Record<string, T[keyof T]> {
  const result: Record<string, T[keyof T]> = {};
  for (const key in obj) {
    result[transformKey(key)] = obj[key];
  }
  return result;
}

/**
 * Transforms object values.
 * @param obj - Object to transform
 * @param transformValue - Value transform function
 * @returns New object with transformed values
 */
export function transformValues<T extends Record<string, any>, U>(
  obj: T,
  transformValue: (value: T[keyof T], key: keyof T) => U
): Record<keyof T, U> {
  const result = {} as Record<keyof T, U>;
  for (const key in obj) {
    result[key] = transformValue(obj[key], key);
  }
  return result;
}

