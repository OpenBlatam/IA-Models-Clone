/**
 * Optimization Utilities
 * Performance optimizations and utilities
 */

import { useCallback, useMemo, useRef } from 'react'

/**
 * Custom hook for debounced callback
 */
export function useDebouncedCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  return useCallback(
    ((...args: Parameters<T>) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }

      timeoutRef.current = setTimeout(() => {
        callback(...args)
      }, delay)
    }) as T,
    [callback, delay]
  )
}

/**
 * Custom hook for throttled callback
 */
export function useThrottledCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T {
  const lastRunRef = useRef<number>(0)

  return useCallback(
    ((...args: Parameters<T>) => {
      const now = Date.now()

      if (now - lastRunRef.current >= delay) {
        lastRunRef.current = now
        callback(...args)
      }
    }) as T,
    [callback, delay]
  )
}

/**
 * Memoized value with equality check
 */
export function useMemoizedValue<T>(
  value: T,
  equalityFn?: (a: T, b: T) => boolean
): T {
  const ref = useRef<T>(value)
  const equality = equalityFn || ((a, b) => a === b)

  if (!equality(ref.current, value)) {
    ref.current = value
  }

  return ref.current
}

/**
 * Intersection Observer hook for lazy loading
 */
export function useIntersectionObserver(
  callback: IntersectionObserverCallback,
  options?: IntersectionObserverInit
) {
  const observerRef = useRef<IntersectionObserver | null>(null)

  return useCallback(
    (element: HTMLElement | null) => {
      if (observerRef.current) {
        observerRef.current.disconnect()
      }

      if (element) {
        observerRef.current = new IntersectionObserver(callback, options)
        observerRef.current.observe(element)
      }
    },
    [callback, options]
  )
}

/**
 * Virtual scrolling utilities
 */
export interface VirtualScrollItem {
  id: string | number
  height?: number
}

export function useVirtualScroll<T extends VirtualScrollItem>(
  items: T[],
  containerHeight: number,
  itemHeight: number = 50,
  overscan: number = 5
) {
  return useMemo(() => {
    const visibleStart = Math.max(0, 0)
    const visibleEnd = Math.min(
      items.length,
      Math.ceil(containerHeight / itemHeight) + overscan
    )

    const visibleItems = items.slice(visibleStart, visibleEnd)
    const totalHeight = items.length * itemHeight
    const offsetY = visibleStart * itemHeight

    return {
      visibleItems,
      totalHeight,
      offsetY,
      startIndex: visibleStart,
      endIndex: visibleEnd
    }
  }, [items, containerHeight, itemHeight, overscan])
}

/**
 * Performance monitoring hook
 */
export function usePerformanceMonitor(componentName: string) {
  const renderStartRef = useRef<number>(0)

  const startRender = useCallback(() => {
    renderStartRef.current = performance.now()
  }, [])

  const endRender = useCallback(() => {
    const duration = performance.now() - renderStartRef.current
    if (duration > 16) {
      // Longer than one frame (16ms)
      console.warn(
        `[Performance] ${componentName} render took ${duration.toFixed(2)}ms`
      )
    }
  }, [componentName])

  return { startRender, endRender }
}

/**
 * Batch state updates
 */
export function useBatchedUpdates() {
  const updatesRef = useRef<(() => void)[]>([])
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  const batchedUpdate = useCallback((update: () => void) => {
    updatesRef.current.push(update)

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    timeoutRef.current = setTimeout(() => {
      const updates = updatesRef.current
      updatesRef.current = []

      updates.forEach(update => update())
    }, 0)
  }, [])

  return batchedUpdate
}

/**
 * Optimized array operations
 */
export function optimizeArrayOperations<T>(array: T[]) {
  return {
    // Memoized filter
    filter: useMemo(
      () => (predicate: (item: T) => boolean) => array.filter(predicate),
      [array]
    ),
    // Memoized map
    map: useMemo(
      () => <U>(mapper: (item: T) => U) => array.map(mapper),
      [array]
    ),
    // Memoized find
    find: useMemo(
      () => (predicate: (item: T) => boolean) => array.find(predicate),
      [array]
    )
  }
}


