/**
 * Hook for event system
 * @module robot-3d-view/hooks/use-events
 */

import { useEffect, useCallback } from 'react';
import { eventManager, type EventType, type EventHandler, type EventSubscription } from '../utils/event-system';

/**
 * Hook for subscribing to events
 * 
 * @param event - Event type
 * @param handler - Event handler
 * @param deps - Dependencies array
 * 
 * @example
 * ```tsx
 * useEvents(Events.CONFIG_CHANGED, (config) => {
 *   console.log('Config changed:', config);
 * }, []);
 * ```
 */
export function useEvents<T = unknown>(
  event: EventType,
  handler: EventHandler<T>,
  deps: unknown[] = []
): void {
  useEffect(() => {
    const subscription = eventManager.on(event, handler);
    return () => {
      subscription.unsubscribe();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [event, ...deps]);
}

/**
 * Hook for emitting events
 * 
 * @returns Function to emit events
 * 
 * @example
 * ```tsx
 * const emit = useEmit();
 * emit(Events.CONFIG_CHANGED, newConfig);
 * ```
 */
export function useEmit() {
  return useCallback(<T = unknown>(event: EventType, data: T) => {
    return eventManager.emit(event, data);
  }, []);
}

/**
 * Hook for subscribing to events once
 * 
 * @param event - Event type
 * @param handler - Event handler
 * @param deps - Dependencies array
 */
export function useEventOnce<T = unknown>(
  event: EventType,
  handler: EventHandler<T>,
  deps: unknown[] = []
): void {
  useEffect(() => {
    const subscription = eventManager.once(event, handler);
    return () => {
      subscription.unsubscribe();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [event, ...deps]);
}



