/**
 * Diff utility functions.
 * Provides helper functions for comparing and diffing data.
 */

/**
 * Diff operation type.
 */
export type DiffOperation = 'add' | 'remove' | 'update' | 'unchanged';

/**
 * Diff result.
 */
export interface DiffResult<T = any> {
  path: string;
  operation: DiffOperation;
  oldValue?: T;
  newValue?: T;
}

/**
 * Calculates diff between two objects.
 */
export function diffObjects<T extends Record<string, any>>(
  oldObj: T,
  newObj: T,
  path: string = ''
): DiffResult[] {
  const diffs: DiffResult[] = [];
  const allKeys = new Set([...Object.keys(oldObj), ...Object.keys(newObj)]);

  for (const key of allKeys) {
    const currentPath = path ? `${path}.${key}` : key;
    const oldValue = oldObj[key];
    const newValue = newObj[key];

    if (!(key in oldObj)) {
      diffs.push({
        path: currentPath,
        operation: 'add',
        newValue,
      });
    } else if (!(key in newObj)) {
      diffs.push({
        path: currentPath,
        operation: 'remove',
        oldValue,
      });
    } else if (typeof oldValue === 'object' && typeof newValue === 'object' && oldValue !== null && newValue !== null && !Array.isArray(oldValue) && !Array.isArray(newValue)) {
      diffs.push(...diffObjects(oldValue, newValue, currentPath));
    } else if (oldValue !== newValue) {
      diffs.push({
        path: currentPath,
        operation: 'update',
        oldValue,
        newValue,
      });
    } else {
      diffs.push({
        path: currentPath,
        operation: 'unchanged',
        oldValue,
        newValue,
      });
    }
  }

  return diffs;
}

/**
 * Calculates diff between two arrays.
 */
export function diffArrays<T>(
  oldArray: T[],
  newArray: T[],
  compareFn?: (a: T, b: T) => boolean
): DiffResult<T[]>[] {
  const diffs: DiffResult<T[]>[] = [];
  const maxLength = Math.max(oldArray.length, newArray.length);

  for (let i = 0; i < maxLength; i++) {
    const oldItem = oldArray[i];
    const newItem = newArray[i];

    if (oldItem === undefined) {
      diffs.push({
        path: `[${i}]`,
        operation: 'add',
        newValue: newItem,
      });
    } else if (newItem === undefined) {
      diffs.push({
        path: `[${i}]`,
        operation: 'remove',
        oldValue: oldItem,
      });
    } else if (compareFn ? !compareFn(oldItem, newItem) : oldItem !== newItem) {
      diffs.push({
        path: `[${i}]`,
        operation: 'update',
        oldValue: oldItem,
        newValue: newItem,
      });
    }
  }

  return diffs;
}

/**
 * Applies diff to an object.
 */
export function applyDiff<T extends Record<string, any>>(
  obj: T,
  diffs: DiffResult[]
): T {
  const result = { ...obj };

  for (const diff of diffs) {
    const keys = diff.path.split('.');
    let current: any = result;

    for (let i = 0; i < keys.length - 1; i++) {
      if (!(keys[i] in current)) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }

    const lastKey = keys[keys.length - 1];

    switch (diff.operation) {
      case 'add':
      case 'update':
        current[lastKey] = diff.newValue;
        break;
      case 'remove':
        delete current[lastKey];
        break;
    }
  }

  return result;
}

