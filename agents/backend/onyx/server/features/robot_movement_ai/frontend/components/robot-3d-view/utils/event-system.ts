/**
 * Advanced event system
 * @module robot-3d-view/utils/event-system
 */

/**
 * Event type
 */
export type EventType = string | symbol;

/**
 * Event handler function
 */
export type EventHandler<T = unknown> = (data: T) => void | Promise<void>;

/**
 * Event subscription
 */
export interface EventSubscription {
  unsubscribe: () => void;
}

/**
 * Event Manager class
 */
export class EventManager {
  private handlers: Map<EventType, Set<EventHandler>> = new Map();
  private onceHandlers: Map<EventType, Set<EventHandler>> = new Map();

  /**
   * Subscribes to an event
   */
  on<T = unknown>(event: EventType, handler: EventHandler<T>): EventSubscription {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler as EventHandler);

    return {
      unsubscribe: () => {
        this.handlers.get(event)?.delete(handler as EventHandler);
      },
    };
  }

  /**
   * Subscribes to an event once
   */
  once<T = unknown>(event: EventType, handler: EventHandler<T>): EventSubscription {
    if (!this.onceHandlers.has(event)) {
      this.onceHandlers.set(event, new Set());
    }
    this.onceHandlers.get(event)!.add(handler as EventHandler);

    return {
      unsubscribe: () => {
        this.onceHandlers.get(event)?.delete(handler as EventHandler);
      },
    };
  }

  /**
   * Unsubscribes from an event
   */
  off<T = unknown>(event: EventType, handler: EventHandler<T>): void {
    this.handlers.get(event)?.delete(handler as EventHandler);
    this.onceHandlers.get(event)?.delete(handler as EventHandler);
  }

  /**
   * Emits an event
   */
  async emit<T = unknown>(event: EventType, data: T): Promise<void> {
    // Call regular handlers
    const handlers = this.handlers.get(event);
    if (handlers) {
      await Promise.all(
        Array.from(handlers).map((handler) => {
          try {
            return Promise.resolve(handler(data));
          } catch (error) {
            console.error(`Error in event handler for ${String(event)}:`, error);
            return Promise.resolve();
          }
        })
      );
    }

    // Call once handlers and remove them
    const onceHandlers = this.onceHandlers.get(event);
    if (onceHandlers) {
      await Promise.all(
        Array.from(onceHandlers).map((handler) => {
          try {
            return Promise.resolve(handler(data));
          } catch (error) {
            console.error(`Error in once event handler for ${String(event)}:`, error);
            return Promise.resolve();
          }
        })
      );
      this.onceHandlers.delete(event);
    }
  }

  /**
   * Removes all handlers for an event
   */
  removeAllListeners(event?: EventType): void {
    if (event) {
      this.handlers.delete(event);
      this.onceHandlers.delete(event);
    } else {
      this.handlers.clear();
      this.onceHandlers.clear();
    }
  }

  /**
   * Gets listener count for an event
   */
  listenerCount(event: EventType): number {
    const handlersCount = this.handlers.get(event)?.size || 0;
    const onceHandlersCount = this.onceHandlers.get(event)?.size || 0;
    return handlersCount + onceHandlersCount;
  }

  /**
   * Gets all event types
   */
  eventNames(): EventType[] {
    const names = new Set<EventType>();
    this.handlers.forEach((_, event) => names.add(event));
    this.onceHandlers.forEach((_, event) => names.add(event));
    return Array.from(names);
  }
}

/**
 * Global event manager instance
 */
export const eventManager = new EventManager();

/**
 * Event types
 */
export const Events = {
  // Configuration events
  CONFIG_CHANGED: 'config:changed',
  CONFIG_LOADED: 'config:loaded',
  CONFIG_SAVED: 'config:saved',

  // Position events
  POSITION_CHANGED: 'position:changed',
  TARGET_CHANGED: 'target:changed',
  TRAJECTORY_UPDATED: 'trajectory:updated',

  // UI events
  UI_OPENED: 'ui:opened',
  UI_CLOSED: 'ui:closed',
  MODAL_OPENED: 'modal:opened',
  MODAL_CLOSED: 'modal:closed',

  // Interaction events
  CLICK: 'interaction:click',
  HOVER: 'interaction:hover',
  KEYBOARD_SHORTCUT: 'interaction:shortcut',

  // Performance events
  PERFORMANCE_WARNING: 'performance:warning',
  FPS_DROPPED: 'performance:fps-dropped',

  // Error events
  ERROR: 'error:occurred',
  WARNING: 'warning:occurred',

  // Lifecycle events
  INITIALIZED: 'lifecycle:initialized',
  DESTROYED: 'lifecycle:destroyed',
} as const;



