/**
 * Custom hook for virtual scrolling
 * Optimizes rendering of large message lists
 */

import { useState, useEffect, useMemo, useRef, useCallback } from 'react'

export interface VirtualScrollOptions {
  itemHeight: number
  containerHeight: number
  overscan?: number
  threshold?: number
}

export interface VirtualScrollState {
  startIndex: number
  endIndex: number
  visibleItems: number[]
  totalHeight: number
  offsetY: number
}

export interface VirtualScrollActions {
  scrollToIndex: (index: number) => void
  scrollToTop: () => void
  scrollToBottom: () => void
  updateContainerHeight: (height: number) => void
}

export function useVirtualization<T>(
  items: T[],
  options: VirtualScrollOptions
): VirtualScrollState & VirtualScrollActions {
  const {
    itemHeight,
    containerHeight,
    overscan = 5,
    threshold = 0.1,
  } = options

  const [scrollTop, setScrollTop] = useState(0)
  const [currentContainerHeight, setCurrentContainerHeight] = useState(containerHeight)
  const containerRef = useRef<HTMLDivElement>(null)

  const totalHeight = items.length * itemHeight
  const visibleCount = Math.ceil(currentContainerHeight / itemHeight)
  
  const startIndex = Math.max(
    0,
    Math.floor(scrollTop / itemHeight) - overscan
  )
  
  const endIndex = Math.min(
    items.length - 1,
    startIndex + visibleCount + overscan * 2
  )

  const visibleItems = useMemo(() => {
    return items.slice(startIndex, endIndex + 1)
  }, [items, startIndex, endIndex])

  const offsetY = startIndex * itemHeight

  const handleScroll = useCallback((e: Event) => {
    const target = e.target as HTMLElement
    setScrollTop(target.scrollTop)
  }, [])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    container.addEventListener('scroll', handleScroll, { passive: true })
    return () => {
      container.removeEventListener('scroll', handleScroll)
    }
  }, [handleScroll])

  const scrollToIndex = useCallback((index: number) => {
    const container = containerRef.current
    if (!container) return

    const targetScrollTop = index * itemHeight
    container.scrollTop = targetScrollTop
    setScrollTop(targetScrollTop)
  }, [itemHeight])

  const scrollToTop = useCallback(() => {
    scrollToIndex(0)
  }, [scrollToIndex])

  const scrollToBottom = useCallback(() => {
    scrollToIndex(items.length - 1)
  }, [scrollToIndex, items.length])

  const updateContainerHeight = useCallback((height: number) => {
    setCurrentContainerHeight(height)
  }, [])

  // Auto-update container height on resize
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        updateContainerHeight(entry.contentRect.height)
      }
    })

    resizeObserver.observe(container)
    return () => {
      resizeObserver.disconnect()
    }
  }, [updateContainerHeight])

  return {
    startIndex,
    endIndex,
    visibleItems: visibleItems.map((_, index) => startIndex + index),
    totalHeight,
    offsetY,
    scrollToIndex,
    scrollToTop,
    scrollToBottom,
    updateContainerHeight,
    containerRef, // Expose ref for container
  } as VirtualScrollState & VirtualScrollActions & { containerRef: React.RefObject<HTMLDivElement> }
}




