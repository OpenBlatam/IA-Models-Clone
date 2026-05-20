/**
 * Custom hook for event emitter.
 * Provides reactive event emitter functionality.
 */

import { useRef, useCallback, useEffect } from 'react';
import { EventEmitter, EventListener } from '../utils/event-emitter';

/**
 * Custom hook for event emitter.
 * Provides reactive event emitter functionality.
 *
 * @returns Event emitter instance
 */
export function useEventEmitter<T extends Record<string, any> = Record<string, any>>() {
  const emitterRef = useRef<EventEmitter<T>>(new EventEmitter<T>());

  useEffect(() => {
    return () => {
      emitterRef.current.removeAllListeners();
    };
  }, []);

  const on = useCallback(
    <K extends keyof T>(event: K, listener: EventListener<T[K]>) => {
      return emitterRef.current.on(event, listener);
    },
    []
  );

  const off = useCallback(
    <K extends keyof T>(event: K, listener: EventListener<T[K]>) => {
      emitterRef.current.off(event, listener);
    },
    []
  );

  const emit = useCallback(<K extends keyof T>(event: K, data: T[K]) => {
    emitterRef.current.emit(event, data);
  }, []);

  const once = useCallback(
    <K extends keyof T>(event: K, listener: EventListener<T[K]>) => {
      emitterRef.current.once(event, listener);
    },
    []
  );

  const removeAllListeners = useCallback(<K extends keyof T>(event?: K) => {
    emitterRef.current.removeAllListeners(event);
  }, []);

  return {
    on,
    off,
    emit,
    once,
    removeAllListeners,
    listenerCount: emitterRef.current.listenerCount.bind(emitterRef.current),
    eventNames: emitterRef.current.eventNames.bind(emitterRef.current),
  };
}

