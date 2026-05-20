/**
 * Proxy utility functions.
 * Provides helper functions for Proxy-based utilities.
 */

/**
 * Creates a deep reactive proxy.
 */
export function reactiveProxy<T extends object>(target: T): T {
  return new Proxy(target, {
    get(obj, prop) {
      const value = Reflect.get(obj, prop);
      if (typeof value === 'object' && value !== null) {
        return reactiveProxy(value);
      }
      return value;
    },
    set(obj, prop, value) {
      const result = Reflect.set(obj, prop, value);
      // Trigger reactivity here if needed
      return result;
    },
  });
}

/**
 * Creates a readonly proxy.
 */
export function readonlyProxy<T extends object>(target: T): T {
  return new Proxy(target, {
    get(obj, prop) {
      return Reflect.get(obj, prop);
    },
    set() {
      console.warn('Cannot set property on readonly object');
      return false;
    },
    deleteProperty() {
      console.warn('Cannot delete property on readonly object');
      return false;
    },
  });
}

/**
 * Creates a validation proxy.
 */
export function validationProxy<T extends object>(
  target: T,
  validator: (prop: string | symbol, value: any) => boolean
): T {
  return new Proxy(target, {
    set(obj, prop, value) {
      if (validator(prop, value)) {
        return Reflect.set(obj, prop, value);
      }
      throw new Error(`Validation failed for property ${String(prop)}`);
    },
  });
}

/**
 * Creates a logging proxy.
 */
export function loggingProxy<T extends object>(
  target: T,
  logger?: (action: string, prop: string | symbol, value?: any) => void
): T {
  const log = logger || console.log;
  return new Proxy(target, {
    get(obj, prop) {
      log('get', prop);
      return Reflect.get(obj, prop);
    },
    set(obj, prop, value) {
      log('set', prop, value);
      return Reflect.set(obj, prop, value);
    },
    deleteProperty(obj, prop) {
      log('delete', prop);
      return Reflect.deleteProperty(obj, prop);
    },
  });
}

