/**
 * Hook useEventListener
 * =====================
 * 
 * Hook para escuchar eventos del DOM de forma segura
 */

import { useEffect, useRef } from 'react'

export type EventTarget = Window | Document | HTMLElement | null

export interface UseEventListenerOptions {
  enabled?: boolean
  capture?: boolean
  once?: boolean
  passive?: boolean
}

/**
 * Hook para escuchar eventos del DOM
 */
export function useEventListener<K extends keyof WindowEventMap>(
  eventName: K,
  handler: (event: WindowEventMap[K]) => void,
  element?: Window | null,
  options?: UseEventListenerOptions
): void

export function useEventListener<K extends keyof DocumentEventMap>(
  eventName: K,
  handler: (event: DocumentEventMap[K]) => void,
  element?: Document | null,
  options?: UseEventListenerOptions
): void

export function useEventListener<K extends keyof HTMLElementEventMap>(
  eventName: K,
  handler: (event: HTMLElementEventMap[K]) => void,
  element: HTMLElement | null,
  options?: UseEventListenerOptions
): void

export function useEventListener(
  eventName: string,
  handler: (event: Event) => void,
  element: EventTarget = typeof window !== 'undefined' ? window : null,
  options: UseEventListenerOptions = {}
): void {
  const { enabled = true, capture, once, passive } = options
  const handlerRef = useRef(handler)

  // Actualizar handler ref cuando cambia
  useEffect(() => {
    handlerRef.current = handler
  }, [handler])

  useEffect(() => {
    if (!enabled || !element) return

    const eventListener = (event: Event) => {
      handlerRef.current(event)
    }

    const optionsObj: AddEventListenerOptions = {}
    if (capture !== undefined) optionsObj.capture = capture
    if (once !== undefined) optionsObj.once = once
    if (passive !== undefined) optionsObj.passive = passive

    element.addEventListener(eventName, eventListener, optionsObj)

    return () => {
      element.removeEventListener(eventName, eventListener, optionsObj)
    }
  }, [eventName, element, enabled, capture, once, passive])
}






