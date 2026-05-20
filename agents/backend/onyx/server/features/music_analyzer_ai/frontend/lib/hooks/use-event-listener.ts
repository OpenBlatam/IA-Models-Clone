/**
 * Custom hook for event listeners.
 * Provides convenient event listener management with automatic cleanup.
 */

import { useEffect, useRef } from 'react';

/**
 * Options for useEventListener hook.
 */
export interface UseEventListenerOptions {
  enabled?: boolean;
  capture?: boolean;
  once?: boolean;
  passive?: boolean;
}

/**
 * Custom hook for event listeners.
 * Adds event listener with automatic cleanup.
 *
 * @param eventName - Event name
 * @param handler - Event handler
 * @param element - Target element (default: window)
 * @param options - Event listener options
 */
export function useEventListener<
  K extends keyof WindowEventMap,
  T extends Window = Window
>(
  eventName: K,
  handler: (event: WindowEventMap[K]) => void,
  element?: T | null,
  options?: UseEventListenerOptions
): void;

export function useEventListener<
  K extends keyof HTMLElementEventMap,
  T extends HTMLElement = HTMLElement
>(
  eventName: K,
  handler: (event: HTMLElementEventMap[K]) => void,
  element: T | null,
  options?: UseEventListenerOptions
): void;

export function useEventListener(
  eventName: string,
  handler: (event: Event) => void,
  element: Window | HTMLElement | null = typeof window !== 'undefined' ? window : null,
  options: UseEventListenerOptions = {}
): void {
  const { enabled = true, capture, once, passive } = options;
  const handlerRef = useRef(handler);

  // Update handler ref when it changes
  useEffect(() => {
    handlerRef.current = handler;
  }, [handler]);

  useEffect(() => {
    if (!enabled || !element) {
      return;
    }

    const eventListener = (event: Event) => {
      handlerRef.current(event);
    };

    const eventOptions: AddEventListenerOptions = {
      capture,
      once,
      passive,
    };

    element.addEventListener(eventName, eventListener, eventOptions);

    return () => {
      element.removeEventListener(eventName, eventListener, eventOptions);
    };
  }, [eventName, element, enabled, capture, once, passive]);
}

