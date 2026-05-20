/**
 * Data transformation utilities
 * Note: Basic array operations (chunk, groupBy, partition, flatten) are in array.ts
 */

// Map array to object
export function mapToObject<T, K extends string | number | symbol, V>(
  array: T[],
  keyFn: (item: T) => K,
  valueFn: (item: T) => V
): Record<K, V> {
  return array.reduce((acc, item) => {
    acc[keyFn(item)] = valueFn(item);
    return acc;
  }, {} as Record<K, V>);
}

// Zip arrays
export function zip<T, U>(array1: T[], array2: U[]): Array<[T, U]> {
  const length = Math.min(array1.length, array2.length);
  const result: Array<[T, U]> = [];

  for (let i = 0; i < length; i++) {
    result.push([array1[i], array2[i]]);
  }

  return result;
}

// Unzip arrays
export function unzip<T, U>(array: Array<[T, U]>): [T[], U[]] {
  return array.reduce(
    (acc, [a, b]) => {
      acc[0].push(a);
      acc[1].push(b);
      return acc;
    },
    [[], []] as [T[], U[]]
  );
}

// Transpose matrix
export function transpose<T>(matrix: T[][]): T[][] {
  if (matrix.length === 0) return [];

  const rows = matrix.length;
  const cols = matrix[0].length;
  const result: T[][] = [];

  for (let i = 0; i < cols; i++) {
    result[i] = [];
    for (let j = 0; j < rows; j++) {
      result[i][j] = matrix[j][i];
    }
  }

  return result;
}

// Re-export common array functions for convenience
export { chunk, groupBy, partition, flatten } from './array';

