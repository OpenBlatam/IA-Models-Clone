/**
 * Custom hook for event emitter
 * 
 * Provides React integration for event emitters
 */

import { useEffect, useRef } from "react";
import { EventEmitter, type EventListener } from "../utils/observable/event-emitter";

/**
 * Custom hook for event emitter
 * 
 * @param emitter - Event emitter instance
 * @param event - Event name
 * @param listener - Event listener
 * 
 * @example
 * ```typescript
 * const emitter = useRef(createEventEmitter()).current;
 * 
 * useEventEmitter(emitter, "agentCreated", (agent) => {
 *   console.log("Agent created:", agent);
 * });
 * 
 * emitter.emit("agentCreated", newAgent);
 * ```
 */
export function useEventEmitter<T extends Record<string, unknown>, K extends keyof T>(
  emitter: EventEmitter<T>,
  event: K,
  listener: EventListener<T[K]>
): void {
  useEffect(() => {
    const unsubscribe = emitter.on(event, listener);
    return unsubscribe;
  }, [emitter, event, listener]);
}




