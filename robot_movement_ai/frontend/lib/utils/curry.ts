/**
 * Function currying utilities
 */

/**
 * Curry a function
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
 * Partial application - bind arguments from the left
 */
export function partial<T extends (...args: any[]) => any>(
  fn: T,
  ...boundArgs: any[]
): (...args: any[]) => ReturnType<T> {
  return (...args: any[]) => fn(...boundArgs, ...args);
}

/**
 * Partial application - bind arguments from the right
 */
export function partialRight<T extends (...args: any[]) => any>(
  fn: T,
  ...boundArgs: any[]
): (...args: any[]) => ReturnType<T> {
  return (...args: any[]) => fn(...args, ...boundArgs);
}

/**
 * Bind specific arguments by position
 */
export function bindArgs<T extends (...args: any[]) => any>(
  fn: T,
  positions: number[],
  values: any[]
): (...args: any[]) => ReturnType<T> {
  return (...args: any[]) => {
    const newArgs: any[] = [];
    let valueIndex = 0;
    let argIndex = 0;

    for (let i = 0; i < fn.length; i++) {
      if (positions.includes(i)) {
        newArgs[i] = values[valueIndex++];
      } else {
        newArgs[i] = args[argIndex++];
      }
    }

    return fn(...newArgs);
  };
}



