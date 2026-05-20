/**
 * Stream utility functions.
 * Provides helper functions for stream processing.
 */

/**
 * Stream processor class.
 */
export class Stream<T> {
  private source: Iterator<T> | Iterable<T>;

  constructor(source: Iterator<T> | Iterable<T>) {
    this.source = source;
  }

  /**
   * Maps over stream.
   */
  map<R>(fn: (value: T) => R): Stream<R> {
    const iterator = this.getIterator();
    return new Stream({
      [Symbol.iterator]: () => ({
        next: () => {
          const result = iterator.next();
          if (result.done) {
            return result;
          }
          return { value: fn(result.value), done: false };
        },
      }),
    });
  }

  /**
   * Filters stream.
   */
  filter(predicate: (value: T) => boolean): Stream<T> {
    const iterator = this.getIterator();
    return new Stream({
      [Symbol.iterator]: () => ({
        next: () => {
          let result = iterator.next();
          while (!result.done && !predicate(result.value)) {
            result = iterator.next();
          }
          return result;
        },
      }),
    });
  }

  /**
   * Takes n items from stream.
   */
  take(count: number): Stream<T> {
    const iterator = this.getIterator();
    let taken = 0;
    return new Stream({
      [Symbol.iterator]: () => ({
        next: () => {
          if (taken >= count) {
            return { value: undefined, done: true };
          }
          taken++;
          return iterator.next();
        },
      }),
    });
  }

  /**
   * Skips n items from stream.
   */
  skip(count: number): Stream<T> {
    const iterator = this.getIterator();
    let skipped = 0;
    return new Stream({
      [Symbol.iterator]: () => ({
        next: () => {
          while (skipped < count) {
            const result = iterator.next();
            if (result.done) {
              return result;
            }
            skipped++;
          }
          return iterator.next();
        },
      }),
    });
  }

  /**
   * Reduces stream to a single value.
   */
  reduce<R>(fn: (acc: R, value: T) => R, initial: R): R {
    let acc = initial;
    const iterator = this.getIterator();
    let result = iterator.next();
    while (!result.done) {
      acc = fn(acc, result.value);
      result = iterator.next();
    }
    return acc;
  }

  /**
   * Collects stream to array.
   */
  toArray(): T[] {
    return Array.from(this.getIterator());
  }

  /**
   * Gets iterator.
   */
  private getIterator(): Iterator<T> {
    if (Symbol.iterator in this.source) {
      return (this.source as Iterable<T>)[Symbol.iterator]();
    }
    return this.source as Iterator<T>;
  }
}

/**
 * Creates a stream from iterable.
 */
export function stream<T>(source: Iterator<T> | Iterable<T>): Stream<T> {
  return new Stream(source);
}

