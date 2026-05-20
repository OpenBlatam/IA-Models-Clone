/**
 * Hook useVirtualList
 * ===================
 * 
 * Hook para listas virtualizadas
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  calculateVirtualItems,
  calculateTotalSize,
  VirtualItem,
  VirtualizationConfig
} from '../utils/virtualizationUtils'

export interface UseVirtualListReturn {
  virtualItems: VirtualItem[]
  totalSize: number
  scrollToIndex: (index: number) => void
  scrollToOffset: (offset: number) => void
  containerRef: React.RefObject<HTMLDivElement>
}

/**
 * Hook para listas virtualizadas
 */
export function useVirtualList(
  config: Omit<VirtualizationConfig, 'containerSize' | 'scrollOffset'>
): UseVirtualListReturn {
  const { itemCount, itemSize, overscan } = config
  const containerRef = useRef<HTMLDivElement>(null)
  const [scrollOffset, setScrollOffset] = useState(0)
  const [containerSize, setContainerSize] = useState(0)

  // Calcular tamaño total
  const totalSize = calculateTotalSize(itemCount, itemSize)

  // Calcular items virtuales
  const virtualItems = calculateVirtualItems({
    itemCount,
    itemSize,
    containerSize,
    overscan,
    scrollOffset
  })

  // Manejar scroll
  const handleScroll = useCallback((e: Event) => {
    const target = e.target as HTMLElement
    setScrollOffset(target.scrollTop)
  }, [])

  // Observar cambios de tamaño
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerSize(entry.contentRect.height)
      }
    })

    resizeObserver.observe(container)
    container.addEventListener('scroll', handleScroll)

    return () => {
      resizeObserver.disconnect()
      container.removeEventListener('scroll', handleScroll)
    }
  }, [handleScroll])

  // Scroll a índice
  const scrollToIndex = useCallback((index: number) => {
    const container = containerRef.current
    if (!container) return

    let offset = 0
    const size = typeof itemSize === 'function' ? itemSize : itemSize

    if (typeof itemSize === 'function') {
      for (let i = 0; i < index; i++) {
        offset += itemSize(i)
      }
    } else {
      offset = index * itemSize
    }

    container.scrollTop = offset
    setScrollOffset(offset)
  }, [itemSize])

  // Scroll a offset
  const scrollToOffset = useCallback((offset: number) => {
    const container = containerRef.current
    if (!container) return

    container.scrollTop = offset
    setScrollOffset(offset)
  }, [])

  return {
    virtualItems,
    totalSize,
    scrollToIndex,
    scrollToOffset,
    containerRef
  }
}






