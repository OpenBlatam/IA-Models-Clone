/**
 * Hook useLazyLoad
 * =================
 * 
 * Hook para lazy loading de recursos
 */

import { useState, useEffect, useRef } from 'react'
import { createLazyObserver } from '../utils/lazyLoadingUtils'

export interface UseLazyLoadOptions {
  root?: Element | null
  rootMargin?: string
  threshold?: number | number[]
  triggerOnce?: boolean
}

/**
 * Hook para lazy loading con Intersection Observer
 */
export function useLazyLoad<T extends HTMLElement = HTMLDivElement>(
  options: UseLazyLoadOptions = {}
): [React.RefObject<T>, boolean] {
  const {
    root = null,
    rootMargin = '50px',
    threshold = 0.01,
    triggerOnce = true
  } = options

  const [isVisible, setIsVisible] = useState(false)
  const elementRef = useRef<T>(null)

  useEffect(() => {
    const element = elementRef.current
    if (!element || (triggerOnce && isVisible)) return

    const observer = createLazyObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            setIsVisible(true)
            if (triggerOnce) {
              observer.unobserve(element)
            }
          } else if (!triggerOnce) {
            setIsVisible(false)
          }
        })
      },
      { root, rootMargin, threshold }
    )

    observer.observe(element)

    return () => {
      observer.disconnect()
    }
  }, [root, rootMargin, threshold, triggerOnce, isVisible])

  return [elementRef, isVisible]
}






