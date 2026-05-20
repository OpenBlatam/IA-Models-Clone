/**
 * Hook useIntersectionObserver
 * ============================
 * 
 * Hook para usar Intersection Observer API
 */

import { useEffect, useRef, useState, RefObject } from 'react'

export interface UseIntersectionObserverOptions extends IntersectionObserverInit {
  triggerOnce?: boolean
}

export interface UseIntersectionObserverResult {
  ref: RefObject<HTMLElement>
  isIntersecting: boolean
  entry: IntersectionObserverEntry | null
}

/**
 * Hook para usar Intersection Observer
 */
export function useIntersectionObserver(
  options: UseIntersectionObserverOptions = {}
): UseIntersectionObserverResult {
  const {
    threshold = 0,
    root = null,
    rootMargin = '0%',
    triggerOnce = false
  } = options

  const [isIntersecting, setIsIntersecting] = useState(false)
  const [entry, setEntry] = useState<IntersectionObserverEntry | null>(null)
  const elementRef = useRef<HTMLElement>(null)

  useEffect(() => {
    const element = elementRef.current
    if (!element) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting)
        setEntry(entry)

        if (entry.isIntersecting && triggerOnce) {
          observer.disconnect()
        }
      },
      {
        threshold,
        root,
        rootMargin
      }
    )

    observer.observe(element)

    return () => {
      observer.disconnect()
    }
  }, [threshold, root, rootMargin, triggerOnce])

  return {
    ref: elementRef,
    isIntersecting,
    entry
  }
}







