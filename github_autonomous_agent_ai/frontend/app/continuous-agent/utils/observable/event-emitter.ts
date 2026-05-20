/**
 * Event emitter implementation
 * 
 * Provides pub/sub pattern for component communication
 */

/**
 * Event listener function
 */
export type EventListener<T = unknown> = (data: T) => void | Promise<void>;

/**
 * Event emitter class
 */
export class EventEmitter<T extends Record<string, unknown> = Record<string, unknown>> {
  private listeners: Map<keyof T, Set<EventListener>> = new Map();

  /**
   * Subscribes to an event
   */
  on<K extends keyof T>(event: K, listener: EventListener<T[K]>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);

    // Return unsubscribe function
    return () => {
      this.off(event, listener);
    };
  }

  /**
   * Unsubscribes from an event
   */
  off<K extends keyof T>(event: K, listener: EventListener<T[K]>): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(listener);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  /**
   * Emits an event
   */
  async emit<K extends keyof T>(event: K, data: T[K]): Promise<void> {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      const promises = Array.from(eventListeners).map((listener) => listener(data));
      await Promise.allSettled(promises);
    }
  }

  /**
   * Subscribes to an event once
   */
  once<K extends keyof T>(event: K, listener: EventListener<T[K]>): void {
    const wrappedListener = async (data: T[K]) => {
      await listener(data);
      this.off(event, wrappedListener);
    };
    this.on(event, wrappedListener);
  }

  /**
   * Removes all listeners for an event
   */
  removeAllListeners<K extends keyof T>(event?: K): void {
    if (event) {
      this.listeners.delete(event);
    } else {
      this.listeners.clear();
    }
  }

  /**
   * Gets listener count for an event
   */
  listenerCount<K extends keyof T>(event: K): number {
    return this.listeners.get(event)?.size ?? 0;
  }

  /**
   * Gets all event names
   */
  eventNames(): Array<keyof T> {
    return Array.from(this.listeners.keys());
  }
}

/**
 * Creates an event emitter
 */
export function createEventEmitter<
  T extends Record<string, unknown> = Record<string, unknown>
>(): EventEmitter<T> {
  return new EventEmitter<T>();
}




