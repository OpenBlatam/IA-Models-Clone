/**
 * Reflection utility functions.
 * Provides helper functions for reflection operations.
 */

/**
 * Gets all property names of an object.
 */
export function getPropertyNames(obj: any): string[] {
  const names: string[] = [];
  let current = obj;

  while (current !== null && current !== Object.prototype) {
    names.push(...Object.getOwnPropertyNames(current));
    current = Object.getPrototypeOf(current);
  }

  return [...new Set(names)];
}

/**
 * Gets all property descriptors.
 */
export function getPropertyDescriptors(obj: any): Record<string, PropertyDescriptor> {
  const descriptors: Record<string, PropertyDescriptor> = {};
  const names = getPropertyNames(obj);

  for (const name of names) {
    const descriptor = Object.getOwnPropertyDescriptor(obj, name);
    if (descriptor) {
      descriptors[name] = descriptor;
    }
  }

  return descriptors;
}

/**
 * Checks if property exists in object or prototype chain.
 */
export function hasProperty(obj: any, prop: string | symbol): boolean {
  return prop in obj;
}

/**
 * Gets property value safely.
 */
export function getProperty<T = any>(obj: any, prop: string | symbol): T | undefined {
  return Reflect.get(obj, prop);
}

/**
 * Sets property value safely.
 */
export function setProperty(obj: any, prop: string | symbol, value: any): boolean {
  return Reflect.set(obj, prop, value);
}

/**
 * Deletes property safely.
 */
export function deleteProperty(obj: any, prop: string | symbol): boolean {
  return Reflect.deleteProperty(obj, prop);
}

/**
 * Defines property with descriptor.
 */
export function defineProperty(
  obj: any,
  prop: string | symbol,
  descriptor: PropertyDescriptor
): boolean {
  return Reflect.defineProperty(obj, prop, descriptor);
}

/**
 * Gets prototype of object.
 */
export function getPrototype(obj: any): object | null {
  return Reflect.getPrototypeOf(obj);
}

/**
 * Sets prototype of object.
 */
export function setPrototype(obj: any, prototype: object | null): boolean {
  return Reflect.setPrototypeOf(obj, prototype);
}

