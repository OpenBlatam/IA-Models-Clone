/**
 * Event utility functions
 * Common event operations
 */

/**
 * Create event emitter
 */
export class EventEmitter<T extends Record<string, unknown[]>> {
  private listeners: {
    [K in keyof T]?: Array<(...args: T[K]) => void>;
  } = {};

  on<K extends keyof T>(event: K, listener: (...args: T[K]) => void): () => void {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event]!.push(listener);

    // Return unsubscribe function
    return () => {
      this.off(event, listener);
    };
  }

  off<K extends keyof T>(event: K, listener: (...args: T[K]) => void): void {
    const listeners = this.listeners[event];
    if (listeners) {
      const index = listeners.indexOf(listener);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  emit<K extends keyof T>(event: K, ...args: T[K]): void {
    const listeners = this.listeners[event];
    if (listeners) {
      listeners.forEach((listener) => listener(...args));
    }
  }

  once<K extends keyof T>(event: K, listener: (...args: T[K]) => void): void {
    const onceWrapper = (...args: T[K]) => {
      listener(...args);
      this.off(event, onceWrapper);
    };
    this.on(event, onceWrapper);
  }

  removeAllListeners<K extends keyof T>(event?: K): void {
    if (event) {
      delete this.listeners[event];
    } else {
      this.listeners = {};
    }
  }
}

/**
 * Create debounced function
 */
export function debounce<T extends unknown[]>(
  fn: (...args: T) => void,
  delay: number
): (...args: T) => void {
  let timeoutId: NodeJS.Timeout | null = null;

  return (...args: T) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    timeoutId = setTimeout(() => {
      fn(...args);
      timeoutId = null;
    }, delay);
  };
}

/**
 * Create throttled function
 */
export function throttle<T extends unknown[]>(
  fn: (...args: T) => void,
  delay: number
): (...args: T) => void {
  let lastCall = 0;

  return (...args: T) => {
    const now = Date.now();

    if (now - lastCall >= delay) {
      lastCall = now;
      fn(...args);
    }
  };
}

/**
 * Create once function
 */
export function once<T extends unknown[]>(
  fn: (...args: T) => void
): (...args: T) => void {
  let called = false;

  return (...args: T) => {
    if (!called) {
      called = true;
      fn(...args);
    }
  };
}

/**
 * Create memoized function
 */
export function memoize<T extends unknown[], R>(
  fn: (...args: T) => R,
  keyFn?: (...args: T) => string
): (...args: T) => R {
  const cache = new Map<string, R>();

  return (...args: T): R => {
    const key = keyFn ? keyFn(...args) : JSON.stringify(args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
}

