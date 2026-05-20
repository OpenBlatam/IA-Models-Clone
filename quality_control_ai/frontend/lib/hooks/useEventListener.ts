import { useEffect, useRef } from 'react';

type EventHandler = (event: Event) => void;

export const useEventListener = (
  eventName: string,
  handler: EventHandler,
  element: Window | Document | HTMLElement | null = window,
  options?: boolean | AddEventListenerOptions
): void => {
  const savedHandler = useRef<EventHandler>();

  useEffect(() => {
    savedHandler.current = handler;
  }, [handler]);

  useEffect(() => {
    if (!element) return;

    const isSupported = element && element.addEventListener;
    if (!isSupported) return;

    const eventListener = (event: Event): void => {
      savedHandler.current?.(event);
    };

    element.addEventListener(eventName, eventListener, options);

    return () => {
      element.removeEventListener(eventName, eventListener, options);
    };
  }, [eventName, element, options]);
};

