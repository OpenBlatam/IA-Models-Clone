/**
 * Event emitter utility.
 * Provides simple event emitter functionality.
 */

/**
 * Event listener function type.
 */
export type EventListener<T = any> = (data: T) => void;

/**
 * Event emitter class.
 */
export class EventEmitter<T extends Record<string, any> = Record<string, any>> {
  private listeners = new Map<keyof T, Set<EventListener>>();

  /**
   * Subscribes to an event.
   */
  on<K extends keyof T>(event: K, listener: EventListener<T[K]>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }

    this.listeners.get(event)!.add(listener as EventListener);

    // Return unsubscribe function
    return () => {
      this.off(event, listener as EventListener);
    };
  }

  /**
   * Unsubscribes from an event.
   */
  off<K extends keyof T>(event: K, listener: EventListener<T[K]>): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(listener as EventListener);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  /**
   * Emits an event.
   */
  emit<K extends keyof T>(event: K, data: T[K]): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach((listener) => {
        listener(data);
      });
    }
  }

  /**
   * Subscribes to an event once.
   */
  once<K extends keyof T>(event: K, listener: EventListener<T[K]>): void {
    const onceListener = (data: T[K]) => {
      listener(data);
      this.off(event, onceListener as EventListener<T[K]>);
    };
    this.on(event, onceListener);
  }

  /**
   * Removes all listeners for an event.
   */
  removeAllListeners<K extends keyof T>(event?: K): void {
    if (event) {
      this.listeners.delete(event);
    } else {
      this.listeners.clear();
    }
  }

  /**
   * Gets listener count for an event.
   */
  listenerCount<K extends keyof T>(event: K): number {
    return this.listeners.get(event)?.size ?? 0;
  }

  /**
   * Gets all event names.
   */
  eventNames(): (keyof T)[] {
    return Array.from(this.listeners.keys());
  }
}

