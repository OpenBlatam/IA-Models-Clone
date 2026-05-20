/**
 * Functional programming utility functions.
 * Provides helper functions for functional programming patterns.
 */

/**
 * Composes multiple functions.
 */
export function compose<T>(...fns: Array<(arg: T) => T>): (arg: T) => T {
  return (arg: T) => fns.reduceRight((acc, fn) => fn(acc), arg);
}

/**
 * Pipes value through multiple functions.
 */
export function pipe<T, R>(
  value: T,
  ...fns: Array<(arg: any) => any>
): R {
  return fns.reduce((acc, fn) => fn(acc), value) as R;
}

/**
 * Curries a function.
 */
export function curry<T extends (...args: any[]) => any>(
  fn: T
): (...args: any[]) => any {
  return function curried(...args: any[]): any {
    if (args.length >= fn.length) {
      return fn(...args);
    }
    return (...nextArgs: any[]) => curried(...args, ...nextArgs);
  };
}

/**
 * Partially applies arguments to a function.
 */
export function partial<T extends (...args: any[]) => any>(
  fn: T,
  ...partialArgs: any[]
): (...remainingArgs: any[]) => ReturnType<T> {
  return (...remainingArgs: any[]) =>
    fn(...partialArgs, ...remainingArgs);
}

/**
 * Creates a function that negates the result of another function.
 */
export function negate<T extends (...args: any[]) => boolean>(
  fn: T
): (...args: Parameters<T>) => boolean {
  return (...args: Parameters<T>) => !fn(...args);
}

/**
 * Creates a function that checks if all predicates are true.
 */
export function all<T>(
  ...predicates: Array<(value: T) => boolean>
): (value: T) => boolean {
  return (value: T) => predicates.every((predicate) => predicate(value));
}

/**
 * Creates a function that checks if any predicate is true.
 */
export function any<T>(
  ...predicates: Array<(value: T) => boolean>
): (value: T) => boolean {
  return (value: T) => predicates.some((predicate) => predicate(value));
}

/**
 * Creates a function that always returns the same value.
 */
export function constant<T>(value: T): () => T {
  return () => value;
}

/**
 * Creates a function that returns the first argument.
 */
export function identity<T>(value: T): T {
  return value;
}

/**
 * Creates a function that ignores arguments and returns undefined.
 */
export function noop(): void {
  // No operation
}

