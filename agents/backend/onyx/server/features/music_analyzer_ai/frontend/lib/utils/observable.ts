/**
 * Observable utility.
 * Provides simple observable pattern implementation.
 */

/**
 * Observer function type.
 */
export type Observer<T> = (value: T) => void;

/**
 * Observable class.
 */
export class Observable<T> {
  private observers = new Set<Observer<T>>();
  private _value: T;

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
   * Sets value and notifies observers.
   */
  set value(newValue: T) {
    if (this._value !== newValue) {
      this._value = newValue;
      this.notify();
    }
  }

  /**
   * Subscribes to value changes.
   */
  subscribe(observer: Observer<T>): () => void {
    this.observers.add(observer);
    // Immediately call observer with current value
    observer(this._value);

    // Return unsubscribe function
    return () => {
      this.observers.delete(observer);
    };
  }

  /**
   * Unsubscribes from value changes.
   */
  unsubscribe(observer: Observer<T>): void {
    this.observers.delete(observer);
  }

  /**
   * Notifies all observers.
   */
  private notify(): void {
    this.observers.forEach((observer) => {
      observer(this._value);
    });
  }

  /**
   * Gets observer count.
   */
  get observerCount(): number {
    return this.observers.size;
  }

  /**
   * Clears all observers.
   */
  clear(): void {
    this.observers.clear();
  }
}

