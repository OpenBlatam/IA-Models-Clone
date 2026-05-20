/**
 * Chain utility functions.
 * Provides helper functions for method chaining.
 */

/**
 * Chainable class.
 */
export class Chain<T> {
  constructor(private value: T) {}

  /**
   * Gets the value.
   */
  valueOf(): T {
    return this.value;
  }

  /**
   * Maps over the value.
   */
  map<R>(fn: (value: T) => R): Chain<R> {
    return new Chain(fn(this.value));
  }

  /**
   * Taps into the value (side effect).
   */
  tap(fn: (value: T) => void): Chain<T> {
    fn(this.value);
    return this;
  }

  /**
   * Filters the value (if array).
   */
  filter(fn: (value: T extends (infer U)[] ? U : never) => boolean): Chain<T> {
    if (Array.isArray(this.value)) {
      return new Chain(this.value.filter(fn as any) as any);
    }
    return this;
  }

  /**
   * Gets property from value (if object).
   */
  get<K extends keyof T>(key: K): Chain<T[K]> {
    if (typeof this.value === 'object' && this.value !== null) {
      return new Chain(this.value[key]);
    }
    throw new Error('Value is not an object');
  }

  /**
   * Sets property on value (if object).
   */
  set<K extends keyof T>(key: K, value: T[K]): Chain<T> {
    if (typeof this.value === 'object' && this.value !== null) {
      return new Chain({ ...this.value, [key]: value } as T);
    }
    throw new Error('Value is not an object');
  }
}

/**
 * Creates a chainable value.
 */
export function chain<T>(value: T): Chain<T> {
  return new Chain(value);
}

