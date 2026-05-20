/**
 * Comparison and diff utilities
 * 
 * Provides functions to compare objects and calculate differences
 */

/**
 * Difference between two values
 */
export interface Diff<T> {
  readonly path: string;
  readonly oldValue: T;
  readonly newValue: T;
  readonly type: "added" | "removed" | "modified" | "unchanged";
}

/**
 * Deep equality check
 */
export function deepEqual<T>(a: T, b: T): boolean {
  if (a === b) {
    return true;
  }

  if (a == null || b == null) {
    return false;
  }

  if (typeof a !== "object" || typeof b !== "object") {
    return false;
  }

  if (Array.isArray(a) !== Array.isArray(b)) {
    return false;
  }

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) {
    return false;
  }

  for (const key of keysA) {
    if (!keysB.includes(key)) {
      return false;
    }

    if (!deepEqual((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key])) {
      return false;
    }
  }

  return true;
}

/**
 * Shallow equality check
 */
export function shallowEqual<T>(a: T, b: T): boolean {
  if (a === b) {
    return true;
  }

  if (a == null || b == null) {
    return false;
  }

  if (typeof a !== "object" || typeof b !== "object") {
    return false;
  }

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) {
    return false;
  }

  for (const key of keysA) {
    if ((a as Record<string, unknown>)[key] !== (b as Record<string, unknown>)[key]) {
      return false;
    }
  }

  return true;
}

/**
 * Calculates deep diff between two objects
 */
export function deepDiff<T extends Record<string, unknown>>(
  oldObj: T,
  newObj: T,
  path: string = ""
): Diff<unknown>[] {
  const diffs: Diff<unknown>[] = [];

  const allKeys = new Set([...Object.keys(oldObj), ...Object.keys(newObj)]);

  for (const key of allKeys) {
    const currentPath = path ? `${path}.${key}` : key;
    const oldValue = oldObj[key];
    const newValue = newObj[key];

    if (!(key in oldObj)) {
      diffs.push({
        path: currentPath,
        oldValue: undefined,
        newValue,
        type: "added",
      });
    } else if (!(key in newObj)) {
      diffs.push({
        path: currentPath,
        oldValue,
        newValue: undefined,
        type: "removed",
      });
    } else if (typeof oldValue === "object" && typeof newValue === "object" && oldValue !== null && newValue !== null && !Array.isArray(oldValue) && !Array.isArray(newValue)) {
      diffs.push(...deepDiff(oldValue as T, newValue as T, currentPath));
    } else if (!deepEqual(oldValue, newValue)) {
      diffs.push({
        path: currentPath,
        oldValue,
        newValue,
        type: "modified",
      });
    }
  }

  return diffs;
}

/**
 * Gets changed fields between two objects
 */
export function getChangedFields<T extends Record<string, unknown>>(
  oldObj: T,
  newObj: T
): Array<keyof T> {
  const diffs = deepDiff(oldObj, newObj);
  return diffs
    .filter((diff) => diff.type !== "unchanged")
    .map((diff) => diff.path.split(".")[0] as keyof T);
}

/**
 * Creates a patch object with only changed fields
 */
export function createPatch<T extends Record<string, unknown>>(
  oldObj: T,
  newObj: T
): Partial<T> {
  const patch: Partial<T> = {};
  const changedFields = getChangedFields(oldObj, newObj);

  for (const field of changedFields) {
    patch[field] = newObj[field];
  }

  return patch;
}

/**
 * Applies a patch to an object
 */
export function applyPatch<T extends Record<string, unknown>>(
  obj: T,
  patch: Partial<T>
): T {
  return { ...obj, ...patch };
}

/**
 * Compares two arrays
 */
export function arrayDiff<T>(
  oldArray: T[],
  newArray: T[],
  compareFn?: (a: T, b: T) => boolean
): {
  readonly added: T[];
  readonly removed: T[];
  readonly modified: Array<{ old: T; new: T }>;
} {
  const compare = compareFn ?? ((a, b) => deepEqual(a, b));
  const added: T[] = [];
  const removed: T[] = [];
  const modified: Array<{ old: T; new: T }> = [];

  const oldMap = new Map(oldArray.map((item, index) => [index, item]));
  const newMap = new Map(newArray.map((item, index) => [index, item]));

  // Find added and modified
  for (const [newIndex, newItem] of newMap) {
    const oldIndex = oldArray.findIndex((oldItem) => compare(oldItem, newItem));
    if (oldIndex === -1) {
      added.push(newItem);
    } else if (!compare(oldArray[oldIndex], newItem)) {
      modified.push({ old: oldArray[oldIndex], new: newItem });
    }
  }

  // Find removed
  for (const [oldIndex, oldItem] of oldMap) {
    const found = newArray.find((newItem) => compare(oldItem, newItem));
    if (!found) {
      removed.push(oldItem);
    }
  }

  return { added, removed, modified };
}




