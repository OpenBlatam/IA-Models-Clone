/**
 * State utility functions
 * Common state operations
 */

/**
 * Create state machine
 */
export class StateMachine<T extends string> {
  private currentState: T;
  private transitions: Map<T, Set<T>> = new Map();
  private onStateChange?: (from: T, to: T) => void;

  constructor(initialState: T, onStateChange?: (from: T, to: T) => void) {
    this.currentState = initialState;
    this.onStateChange = onStateChange;
  }

  addTransition(from: T, to: T): void {
    if (!this.transitions.has(from)) {
      this.transitions.set(from, new Set());
    }
    this.transitions.get(from)!.add(to);
  }

  canTransition(to: T): boolean {
    const allowed = this.transitions.get(this.currentState);
    return allowed ? allowed.has(to) : false;
  }

  transition(to: T): boolean {
    if (this.canTransition(to)) {
      const from = this.currentState;
      this.currentState = to;
      this.onStateChange?.(from, to);
      return true;
    }
    return false;
  }

  getState(): T {
    return this.currentState;
  }

  reset(to: T): void {
    this.currentState = to;
  }
}

/**
 * Create observable state
 */
export class ObservableState<T> {
  private value: T;
  private listeners: Set<(value: T) => void> = new Set();

  constructor(initialValue: T) {
    this.value = initialValue;
  }

  get(): T {
    return this.value;
  }

  set(value: T): void {
    if (this.value !== value) {
      this.value = value;
      this.listeners.forEach((listener) => listener(value));
    }
  }

  subscribe(listener: (value: T) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  unsubscribe(listener: (value: T) => void): void {
    this.listeners.delete(listener);
  }
}

/**
 * Create cached state
 */
export class CachedState<T> {
  private value: T | null = null;
  private timestamp: number = 0;
  private ttl: number;

  constructor(ttl: number = 60000) {
    this.ttl = ttl;
  }

  get(): T | null {
    if (this.value !== null && Date.now() - this.timestamp < this.ttl) {
      return this.value;
    }
    return null;
  }

  set(value: T): void {
    this.value = value;
    this.timestamp = Date.now();
  }

  clear(): void {
    this.value = null;
    this.timestamp = 0;
  }

  isExpired(): boolean {
    return this.value === null || Date.now() - this.timestamp >= this.ttl;
  }
}

