/**
 * Hook for managing 3D view events
 * @module robot-3d-view/hooks/use-3d-events
 */

import { useCallback, useRef } from 'react';
import type { Position3D } from '../types';
import { createViewEvent, createDebouncedHandler, createThrottledHandler, type ViewEventHandler, type ViewEventType } from '../utils/event-handlers';

/**
 * Options for 3D events
 */
interface EventOptions {
  /** Enable click events */
  enableClick?: boolean;
  /** Enable hover events */
  enableHover?: boolean;
  /** Debounce delay for hover events */
  hoverDebounce?: number;
  /** Throttle limit for click events */
  clickThrottle?: number;
  /** Callback for all events */
  onEvent?: ViewEventHandler;
}

/**
 * Hook for managing 3D view events
 * 
 * Provides event handlers for 3D interactions with debouncing and throttling.
 * 
 * @param options - Event configuration
 * @returns Event handlers
 * 
 * @example
 * ```tsx
 * const { handleClick, handleHover } = use3DEvents({
 *   onEvent: (event) => console.log('Event:', event),
 * });
 * ```
 */
export function use3DEvents(options: EventOptions = {}) {
  const {
    enableClick = true,
    enableHover = true,
    hoverDebounce = 100,
    clickThrottle = 200,
    onEvent,
  } = options;

  const eventHandlersRef = useRef<Map<ViewEventType, ViewEventHandler>>(new Map());

  const createHandler = useCallback(
    (type: ViewEventType, customHandler?: ViewEventHandler) => {
      if (!eventHandlersRef.current.has(type)) {
        let handler: ViewEventHandler = (event) => {
          onEvent?.(event);
          customHandler?.(event);
        };

        // Apply throttling/debouncing based on event type
        if (type === 'hover') {
          handler = createDebouncedHandler(handler, hoverDebounce);
        } else if (type === 'click') {
          handler = createThrottledHandler(handler, clickThrottle);
        }

        eventHandlersRef.current.set(type, handler);
      }

      return eventHandlersRef.current.get(type)!;
    },
    [onEvent, hoverDebounce, clickThrottle]
  );

  const handleClick = useCallback(
    (position: Position3D, target?: string) => {
      if (!enableClick) return;
      const handler = createHandler('click');
      handler(createViewEvent('click', position, target));
    },
    [enableClick, createHandler]
  );

  const handleHover = useCallback(
    (position: Position3D, isEntering: boolean, target?: string) => {
      if (!enableHover) return;
      const handler = createHandler('hover');
      if (isEntering) {
        handler(createViewEvent('hover', position, target));
      }
    },
    [enableHover, createHandler]
  );

  return {
    handleClick,
    handleHover,
  };
}



