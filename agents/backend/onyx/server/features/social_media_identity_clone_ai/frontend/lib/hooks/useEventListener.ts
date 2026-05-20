import { useEffect, useRef } from 'react';

type EventTarget = Window | Document | HTMLElement | null;

export const useEventListener = <K extends keyof WindowEventMap>(
  eventName: K,
  handler: (event: WindowEventMap[K]) => void,
  element?: EventTarget,
  options?: boolean | AddEventListenerOptions
): void => {
  const savedHandler = useRef<typeof handler>();

  useEffect(() => {
    savedHandler.current = handler;
  }, [handler]);

  useEffect(() => {
    const targetElement: EventTarget = element || window;

    if (!targetElement || !targetElement.addEventListener) {
      return;
    }

    const eventListener = (event: Event): void => {
      savedHandler.current?.(event as WindowEventMap[K]);
    };

    targetElement.addEventListener(eventName, eventListener, options);

    return () => {
      targetElement.removeEventListener(eventName, eventListener, options);
    };
  }, [eventName, element, options]);
};



