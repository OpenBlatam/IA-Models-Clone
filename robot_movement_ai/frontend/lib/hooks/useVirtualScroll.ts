import { useState, useMemo, useRef, useEffect, useCallback } from 'react';

export interface UseVirtualScrollOptions {
  itemHeight: number;
  overscan?: number;
  containerHeight?: number;
}

/**
 * Hook for virtual scrolling
 */
export function useVirtualScroll<T>(
  items: T[],
  options: UseVirtualScrollOptions
) {
  const { itemHeight, overscan = 3, containerHeight } = options;
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const visibleRange = useMemo(() => {
    const startIndex = Math.floor(scrollTop / itemHeight);
    const endIndex = Math.min(
      startIndex + Math.ceil((containerHeight || 400) / itemHeight) + overscan,
      items.length
    );
    return {
      startIndex: Math.max(0, startIndex - overscan),
      endIndex,
    };
  }, [scrollTop, itemHeight, containerHeight, overscan, items.length]);

  const visibleItems = useMemo(() => {
    return items.slice(visibleRange.startIndex, visibleRange.endIndex);
  }, [items, visibleRange]);

  const totalHeight = items.length * itemHeight;
  const offsetY = visibleRange.startIndex * itemHeight;

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScrollEvent = (e: Event) => {
      const target = e.target as HTMLDivElement;
      setScrollTop(target.scrollTop);
    };

    container.addEventListener('scroll', handleScrollEvent);
    return () => container.removeEventListener('scroll', handleScrollEvent);
  }, []);

  return {
    containerRef,
    visibleItems,
    totalHeight,
    offsetY,
    handleScroll,
    startIndex: visibleRange.startIndex,
    endIndex: visibleRange.endIndex,
  };
}



