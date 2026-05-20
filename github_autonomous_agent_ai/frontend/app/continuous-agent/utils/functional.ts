type AnyFunction = (...args: any[]) => any;

export const compose = <T extends AnyFunction[]>(
  ...fns: T
): ((...args: Parameters<T[0]>) => ReturnType<T[T["length"] extends number ? T["length"] extends 1 ? 0 : never : never]>>) => {
  return ((...args: any[]) => {
    return fns.reduceRight((acc, fn) => fn(acc), args[0]);
  }) as any;
};

export const pipe = <T extends AnyFunction[]>(
  ...fns: T
): ((...args: Parameters<T[0]>) => ReturnType<T[T["length"] extends number ? T["length"] extends 1 ? 0 : never : never]>>) => {
  return ((...args: any[]) => {
    return fns.reduce((acc, fn) => fn(acc), args[0]);
  }) as any;
};

export const curry = <T extends AnyFunction>(
  fn: T,
  arity?: number
): any => {
  const curriedArity = arity ?? fn.length;
  
  const curried = (...args: any[]): any => {
    if (args.length >= curriedArity) {
      return fn(...args);
    }
    return (...nextArgs: any[]) => curried(...args, ...nextArgs);
  };
  
  return curried;
};

export const memoize = <T extends AnyFunction>(
  fn: T,
  keyGenerator?: (...args: Parameters<T>) => string
): T => {
  const cache = new Map<string, ReturnType<T>>();
  
  const memoized = ((...args: Parameters<T>): ReturnType<T> => {
    const key = keyGenerator
      ? keyGenerator(...args)
      : JSON.stringify(args);
    
    if (cache.has(key)) {
      return cache.get(key)!;
    }
    
    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as T;
  
  return memoized;
};

export const once = <T extends AnyFunction>(fn: T): T => {
  let called = false;
  let result: ReturnType<T>;
  
  return ((...args: Parameters<T>): ReturnType<T> => {
    if (!called) {
      called = true;
      result = fn(...args);
    }
    return result;
  }) as T;
};

export const debounceFn = <T extends AnyFunction>(
  fn: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout | null = null;
  
  return (...args: Parameters<T>): void => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      fn(...args);
    }, delay);
  };
};

export const throttleFn = <T extends AnyFunction>(
  fn: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let lastCall = 0;
  
  return (...args: Parameters<T>): void => {
    const now = Date.now();
    if (now - lastCall >= delay) {
      lastCall = now;
      fn(...args);
    }
  };
};

export const guardArgs = <T extends AnyFunction>(
  fn: T
): ((...args: Parameters<T>) => ReturnType<T> | undefined) => {
  return (...args: Parameters<T>): ReturnType<T> | undefined => {
    if (args.every((arg) => arg != null)) {
      return fn(...args);
    }
    return undefined;
  };
};

export const defaultArgs = <T extends AnyFunction>(
  fn: T,
  defaults: Partial<Parameters<T>[0]>
): T => {
  return ((...args: Parameters<T>): ReturnType<T> => {
    const merged = { ...defaults, ...args[0] };
    return fn(merged as Parameters<T>[0], ...args.slice(1));
  }) as T;
};

export const tap = <T>(value: T, fn: (value: T) => void): T => {
  fn(value);
  return value;
};

export const identity = <T>(value: T): T => value;

export const constant = <T>(value: T): () => T => () => value;

export const noop = (): void => {};





