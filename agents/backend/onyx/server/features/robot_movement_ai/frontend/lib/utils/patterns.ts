/**
 * Common design patterns utilities
 */

/**
 * Singleton pattern
 */
export function createSingleton<T>(factory: () => T): () => T {
  let instance: T | null = null;

  return () => {
    if (instance === null) {
      instance = factory();
    }
    return instance;
  };
}

/**
 * Factory pattern
 */
export function createFactory<T, P extends any[]>(
  factoryFn: (...args: P) => T
): (...args: P) => T {
  return (...args: P) => factoryFn(...args);
}

/**
 * Observer pattern
 */
export class Observable<T> {
  private observers: Set<(value: T) => void> = new Set();

  subscribe(observer: (value: T) => void): () => void {
    this.observers.add(observer);
    return () => this.observers.delete(observer);
  }

  notify(value: T): void {
    this.observers.forEach((observer) => observer(value));
  }

  clear(): void {
    this.observers.clear();
  }
}

/**
 * Pub/Sub pattern
 */
export class EventEmitter {
  private events: Map<string, Set<(...args: any[]) => void>> = new Map();

  on(event: string, handler: (...args: any[]) => void): () => void {
    if (!this.events.has(event)) {
      this.events.set(event, new Set());
    }
    this.events.get(event)!.add(handler);
    return () => this.off(event, handler);
  }

  off(event: string, handler: (...args: any[]) => void): void {
    this.events.get(event)?.delete(handler);
  }

  emit(event: string, ...args: any[]): void {
    this.events.get(event)?.forEach((handler) => handler(...args));
  }

  once(event: string, handler: (...args: any[]) => void): () => void {
    const onceHandler = (...args: any[]) => {
      handler(...args);
      this.off(event, onceHandler);
    };
    return this.on(event, onceHandler);
  }

  clear(event?: string): void {
    if (event) {
      this.events.delete(event);
    } else {
      this.events.clear();
    }
  }
}

/**
 * Strategy pattern
 */
export function createStrategy<T, R>(
  strategies: Record<string, (input: T) => R>
) {
  return (strategyName: string, input: T): R => {
    const strategy = strategies[strategyName];
    if (!strategy) {
      throw new Error(`Strategy "${strategyName}" not found`);
    }
    return strategy(input);
  };
}

/**
 * Chain of responsibility pattern
 */
export function createChain<T>(
  handlers: Array<(input: T, next: () => T) => T>
): (input: T) => T {
  return (input: T) => {
    let index = 0;

    const next = (): T => {
      if (index >= handlers.length) {
        return input;
      }
      const handler = handlers[index++];
      return handler(input, next);
    };

    return next();
  };
}



