/**
 * Type guard utilities
 */

/**
 * Type guard for defined values
 */
export function isDefined<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

/**
 * Type guard for strings
 */
export function isString(value: unknown): value is string {
  return typeof value === 'string';
}

/**
 * Type guard for numbers
 */
export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value);
}

/**
 * Type guard for booleans
 */
export function isBoolean(value: unknown): value is boolean {
  return typeof value === 'boolean';
}

/**
 * Type guard for objects (not arrays or null)
 */
export function isObject(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Type guard for arrays
 */
export function isArray(value: unknown): value is any[] {
  return Array.isArray(value);
}

/**
 * Type guard for functions
 */
export function isFunction(value: unknown): value is Function {
  return typeof value === 'function';
}

/**
 * Type guard for dates
 */
export function isDate(value: unknown): value is Date {
  return value instanceof Date && !isNaN(value.getTime());
}

/**
 * Type guard for promises
 */
export function isPromise<T>(value: unknown): value is Promise<T> {
  return (
    value !== null &&
    typeof value === 'object' &&
    'then' in value &&
    typeof (value as any).then === 'function'
  );
}

/**
 * Type guard for error objects
 */
export function isError(value: unknown): value is Error {
  return value instanceof Error;
}

/**
 * Type guard for empty values
 */
export function isEmpty(value: unknown): boolean {
  if (value === null || value === undefined) return true;
  if (typeof value === 'string') return value.length === 0;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value).length === 0;
  return false;
}

/**
 * Assert value is defined, throw if not
 */
export function assertDefined<T>(
  value: T | null | undefined,
  message?: string
): asserts value is T {
  if (value === null || value === undefined) {
    throw new Error(message || 'Value is not defined');
  }
}

/**
 * Assert value is truthy, throw if not
 */
export function assertTruthy<T>(
  value: T | null | undefined | false | 0 | '',
  message?: string
): asserts value is T {
  if (!value) {
    throw new Error(message || 'Value is not truthy');
  }
}



