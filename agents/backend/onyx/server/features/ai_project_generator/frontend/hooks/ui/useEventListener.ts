import { useEffect, useRef } from 'react'

type EventHandler = (event: Event) => void

export const useEventListener = <T extends HTMLElement = HTMLElement>(
  eventName: string,
  handler: EventHandler,
  element?: T | Window | Document | null,
  options?: boolean | AddEventListenerOptions
) => {
  const savedHandler = useRef<EventHandler>()

  useEffect(() => {
    savedHandler.current = handler
  }, [handler])

  useEffect(() => {
    const targetElement = element || window
    if (!targetElement || !targetElement.addEventListener) {
      return
    }

    const eventListener = (event: Event) => {
      savedHandler.current?.(event)
    }

    targetElement.addEventListener(eventName, eventListener, options)

    return () => {
      targetElement.removeEventListener(eventName, eventListener, options)
    }
  }, [eventName, element, options])
}

