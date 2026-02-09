/**
 * Event handler utilities for 3D interactions
 * @module robot-3d-view/utils/event-handlers
 */

import type { Position3D } from '../types';

/**
 * Event types for 3D view
 */
export type ViewEventType = 'click' | 'hover' | 'select' | 'deselect';

/**
 * Event data structure
 */
export interface ViewEvent {
  type: ViewEventType;
  position: Position3D;
  timestamp: number;
  target?: string;
}

/**
 * Event handler function type
 */
export type ViewEventHandler = (event: ViewEvent) => void;

/**
 * Creates a debounced event handler
 * 
 * @param handler - Event handler function
 * @param delay - Debounce delay in milliseconds
 * @returns Debounced handler
 */
export function createDebouncedHandler(
  handler: ViewEventHandler,
  delay: number = 100
): ViewEventHandler {
  let timeoutId: NodeJS.Timeout;

  return (event: ViewEvent) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => handler(event), delay);
  };
}

/**
 * Creates a throttled event handler
 * 
 * @param handler - Event handler function
 * @param limit - Throttle limit in milliseconds
 * @returns Throttled handler
 */
export function createThrottledHandler(
  handler: ViewEventHandler,
  limit: number = 100
): ViewEventHandler {
  let lastCall = 0;

  return (event: ViewEvent) => {
    const now = Date.now();
    if (now - lastCall >= limit) {
      lastCall = now;
      handler(event);
    }
  };
}

/**
 * Creates a view event
 * 
 * @param type - Event type
 * @param position - Position where event occurred
 * @param target - Optional target identifier
 * @returns View event object
 */
export function createViewEvent(
  type: ViewEventType,
  position: Position3D,
  target?: string
): ViewEvent {
  return {
    type,
    position,
    timestamp: Date.now(),
    target,
  };
}



