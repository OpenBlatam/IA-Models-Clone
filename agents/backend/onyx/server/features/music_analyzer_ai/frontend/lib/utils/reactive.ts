/**
 * Reactive utility functions.
 * Provides helper functions for reactive programming.
 */

/**
 * Reactive value class.
 */
export class Reactive<T> {
  private _value: T;
  private subscribers = new Set<(value: T) => void>();

  constructor(initialValue: T) {
    this._value = initialValue;
  }

  /**
   * Gets current value.
   */
  get value(): T {
    return this._value;
  }

  /**
   * Sets value and notifies subscribers.
   */
  set value(newValue: T) {
    if (this._value !== newValue) {
      this._value = newValue;
      this.notify();
    }
  }

  /**
   * Updates value using function.
   */
  update(fn: (value: T) => T): void {
    this.value = fn(this._value);
  }

  /**
   * Subscribes to value changes.
   */
  subscribe(fn: (value: T) => void): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  /**
   * Notifies all subscribers.
   */
  private notify(): void {
    this.subscribers.forEach((fn) => fn(this._value));
  }

  /**
   * Creates a computed reactive value.
   */
  map<R>(fn: (value: T) => R): Reactive<R> {
    const computed = new Reactive(fn(this._value));
    this.subscribe((value) => {
      computed.value = fn(value);
    });
    return computed;
  }
}

/**
 * Creates a reactive value.
 */
export function reactive<T>(initialValue: T): Reactive<T> {
  return new Reactive(initialValue);
}

/**
 * Creates a computed reactive value.
 */
export function computed<T, R>(
  source: Reactive<T>,
  fn: (value: T) => R
): Reactive<R> {
  return source.map(fn);
}

