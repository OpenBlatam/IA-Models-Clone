/**
 * Comparison utility functions
 * Common comparison operations
 */

/**
 * Compare two values
 */
export function compare<T>(a: T, b: T): number {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

/**
 * Compare two values (reverse order)
 */
export function compareReverse<T>(a: T, b: T): number {
  return compare(b, a);
}

/**
 * Compare by key
 */
export function compareBy<T, K extends keyof T>(
  key: K,
  order: 'asc' | 'desc' = 'asc'
) {
  return (a: T, b: T): number => {
    const aVal = a[key];
    const bVal = b[key];

    if (aVal < bVal) return order === 'asc' ? -1 : 1;
    if (aVal > bVal) return order === 'asc' ? 1 : -1;
    return 0;
  };
}

/**
 * Compare by multiple keys
 */
export function compareByMultiple<T>(
  comparators: Array<(a: T, b: T) => number>
) {
  return (a: T, b: T): number => {
    for (const comparator of comparators) {
      const result = comparator(a, b);
      if (result !== 0) {
        return result;
      }
    }
    return 0;
  };
}

/**
 * Check if values are equal (deep comparison)
 */
export function deepEqual<T>(a: T, b: T): boolean {
  if (a === b) return true;

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
 * Check if value is in array
 */
export function isInArray<T>(value: T, array: T[]): boolean {
  return array.includes(value);
}

/**
 * Check if all values are equal
 */
export function allEqual<T>(...values: T[]): boolean {
  if (values.length <= 1) return true;

  const first = values[0];
  return values.every((value) => deepEqual(value, first));
}

/**
 * Find differences between two objects
 */
export function findDifferences<T extends Record<string, unknown>>(
  obj1: T,
  obj2: T
): Array<{ key: string; oldValue: unknown; newValue: unknown }> {
  const differences: Array<{
    key: string;
    oldValue: unknown;
    newValue: unknown;
  }> = [];

  const allKeys = new Set([...Object.keys(obj1), ...Object.keys(obj2)]);

  for (const key of allKeys) {
    const val1 = obj1[key];
    const val2 = obj2[key];

    if (!deepEqual(val1, val2)) {
      differences.push({
        key,
        oldValue: val1,
        newValue: val2,
      });
    }
  }

  return differences;
}

