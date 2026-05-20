/**
 * Object Utility Functions
 * Utility functions for object manipulation
 */

/**
 * Omits specified keys from an object
 * @param obj - Object to omit keys from
 * @param keys - Keys to omit
 * @returns New object without specified keys
 */
export const omit = <T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> => {
  const result = { ...obj };
  keys.forEach((key) => {
    delete result[key];
  });
  return result;
};

/**
 * Picks specified keys from an object
 * @param obj - Object to pick keys from
 * @param keys - Keys to pick
 * @returns New object with only specified keys
 */
export const pick = <T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> => {
  const result = {} as Pick<T, K>;
  keys.forEach((key) => {
    if (key in obj) {
      result[key] = obj[key];
    }
  });
  return result;
};

/**
 * Checks if an object is empty
 * @param obj - Object to check
 * @returns True if object is empty
 */
export const isEmpty = (obj: Record<string, unknown>): boolean => {
  return Object.keys(obj).length === 0;
};

/**
 * Deep merges two objects
 * @param target - Target object
 * @param source - Source object
 * @returns Merged object
 */
export const deepMerge = <T extends Record<string, unknown>>(
  target: T,
  source: Partial<T>
): T => {
  const output = { ...target };
  
  if (isObject(target) && isObject(source)) {
    Object.keys(source).forEach((key) => {
      const sourceValue = source[key];
      const targetValue = target[key];
      
      if (isObject(sourceValue) && isObject(targetValue)) {
        output[key] = deepMerge(targetValue as Record<string, unknown>, sourceValue as Record<string, unknown>) as T[Extract<keyof T, string>];
      } else {
        output[key] = sourceValue as T[Extract<keyof T, string>];
      }
    });
  }
  
  return output;
};

/**
 * Checks if a value is an object
 * @param value - Value to check
 * @returns True if value is an object
 */
const isObject = (value: unknown): value is Record<string, unknown> => {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
};

/**
 * Gets nested value from object using dot notation
 * @param obj - Object to get value from
 * @param path - Dot notation path (e.g., 'user.profile.name')
 * @returns Value at path or undefined
 */
export const get = <T,>(obj: Record<string, unknown>, path: string): T | undefined => {
  const keys = path.split('.');
  let result: unknown = obj;
  
  for (const key of keys) {
    if (result === null || result === undefined) {
      return undefined;
    }
    result = (result as Record<string, unknown>)[key];
  }
  
  return result as T | undefined;
};

/**
 * Sets nested value in object using dot notation
 * @param obj - Object to set value in
 * @param path - Dot notation path
 * @param value - Value to set
 * @returns New object with value set
 */
export const set = <T extends Record<string, unknown>>(
  obj: T,
  path: string,
  value: unknown
): T => {
  const keys = path.split('.');
  const result = { ...obj };
  let current: Record<string, unknown> = result;
  
  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (!(key in current) || !isObject(current[key])) {
      current[key] = {};
    }
    current = current[key] as Record<string, unknown>;
  }
  
  current[keys[keys.length - 1]] = value;
  return result;
};


