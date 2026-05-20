/**
 * Iterator utility functions.
 * Provides helper functions for working with iterators.
 */

/**
 * Creates an iterator from a generator function.
 */
export function* range(start: number, end: number, step: number = 1): Generator<number> {
  for (let i = start; i < end; i += step) {
    yield i;
  }
}

/**
 * Creates an infinite iterator.
 */
export function* infinite<T>(value: T): Generator<T> {
  while (true) {
    yield value;
  }
}

/**
 * Creates an iterator that repeats a value.
 */
export function* repeat<T>(value: T, count: number): Generator<T> {
  for (let i = 0; i < count; i++) {
    yield value;
  }
}

/**
 * Creates an iterator that cycles through values.
 */
export function* cycle<T>(values: T[]): Generator<T> {
  let index = 0;
  while (true) {
    yield values[index];
    index = (index + 1) % values.length;
  }
}

/**
 * Maps over an iterator.
 */
export function* mapIterator<T, R>(
  iterator: Iterator<T>,
  fn: (value: T) => R
): Generator<R> {
  let result = iterator.next();
  while (!result.done) {
    yield fn(result.value);
    result = iterator.next();
  }
}

/**
 * Filters an iterator.
 */
export function* filterIterator<T>(
  iterator: Iterator<T>,
  predicate: (value: T) => boolean
): Generator<T> {
  let result = iterator.next();
  while (!result.done) {
    if (predicate(result.value)) {
      yield result.value;
    }
    result = iterator.next();
  }
}

/**
 * Takes n items from an iterator.
 */
export function* takeIterator<T>(
  iterator: Iterator<T>,
  count: number
): Generator<T> {
  let taken = 0;
  let result = iterator.next();
  while (!result.done && taken < count) {
    yield result.value;
    taken++;
    result = iterator.next();
  }
}

/**
 * Skips n items from an iterator.
 */
export function* skipIterator<T>(
  iterator: Iterator<T>,
  count: number
): Generator<T> {
  let skipped = 0;
  let result = iterator.next();
  while (!result.done && skipped < count) {
    skipped++;
    result = iterator.next();
  }
  while (!result.done) {
    yield result.value;
    result = iterator.next();
  }
}

/**
 * Zips multiple iterators together.
 */
export function* zip<T extends readonly unknown[]>(
  ...iterators: { [K in keyof T]: Iterator<T[K]> }
): Generator<T> {
  while (true) {
    const results = iterators.map((it) => it.next());
    if (results.some((r) => r.done)) {
      break;
    }
    yield results.map((r) => r.value) as T;
  }
}

/**
 * Converts iterator to array.
 */
export function iteratorToArray<T>(iterator: Iterator<T>): T[] {
  const array: T[] = [];
  let result = iterator.next();
  while (!result.done) {
    array.push(result.value);
    result = iterator.next();
  }
  return array;
}

