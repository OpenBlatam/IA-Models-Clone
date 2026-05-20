/**
 * Type checking utilities
 */

// Check if value is defined
export function isDefined<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

// Check if value is null
export function isNull(value: any): value is null {
  return value === null;
}

// Check if value is undefined
export function isUndefined(value: any): value is undefined {
  return value === undefined;
}

// Check if value is null or undefined
export function isNullOrUndefined(value: any): value is null | undefined {
  return value === null || value === undefined;
}

// Check if value is string
export function isString(value: any): value is string {
  return typeof value === 'string';
}

// Check if value is number
export function isNumber(value: any): value is number {
  return typeof value === 'number' && !isNaN(value);
}

// Check if value is boolean
export function isBoolean(value: any): value is boolean {
  return typeof value === 'boolean';
}

// Check if value is object
export function isObject(value: any): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

// Check if value is array
export function isArray(value: any): value is any[] {
  return Array.isArray(value);
}

// Check if value is function
export function isFunction(value: any): value is Function {
  return typeof value === 'function';
}

// Check if value is date
export function isDate(value: any): value is Date {
  return value instanceof Date && !isNaN(value.getTime());
}

// Check if value is empty
export function isEmpty(value: any): boolean {
  if (value === null || value === undefined) return true;
  if (typeof value === 'string') return value.length === 0;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value).length === 0;
  return false;
}

// Get type name
export function getType(value: any): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}



