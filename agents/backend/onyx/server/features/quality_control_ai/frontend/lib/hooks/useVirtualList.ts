import { useState, useEffect, useRef, useMemo, useCallback } from 'react';

interface UseVirtualListOptions {
  itemHeight: number | ((index: number) => number);
  containerHeight: number;
  overscan?: number;
}

export const useVirtualList = <T,>(
  items: T[],
  options: UseVirtualListOptions
) => {
  const { itemHeight, containerHeight, overscan = 3 } = options;
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const getItemHeight = useCallback(
    (index: number): number => {
      return typeof itemHeight === 'function' ? itemHeight(index) : itemHeight;
    },
    [itemHeight]
  );

  const totalHeight = useMemo(() => {
    return items.reduce((sum, _, index) => sum + getItemHeight(index), 0);
  }, [items, getItemHeight]);

  const visibleRange = useMemo(() => {
    let start = 0;
    let end = 0;
    let currentTop = 0;

    for (let i = 0; i < items.length; i++) {
      const height = getItemHeight(i);
      const itemBottom = currentTop + height;

      if (itemBottom < scrollTop) {
        start = i + 1;
      } else if (currentTop <= scrollTop + containerHeight) {
        end = i;
      } else {
        break;
      }

      currentTop = itemBottom;
    }

    return {
      start: Math.max(0, start - overscan),
      end: Math.min(items.length - 1, end + overscan),
    };
  }, [items, scrollTop, containerHeight, overscan, getItemHeight]);

  const visibleItems = useMemo(() => {
    return items.slice(visibleRange.start, visibleRange.end + 1).map((item, index) => ({
      item,
      index: visibleRange.start + index,
    }));
  }, [items, visibleRange]);

  const offsetY = useMemo(() => {
    let offset = 0;
    for (let i = 0; i < visibleRange.start; i++) {
      offset += getItemHeight(i);
    }
    return offset;
  }, [visibleRange.start, getItemHeight]);

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      setScrollTop(container.scrollTop);
    }
  }, []);

  return {
    containerRef,
    visibleItems,
    totalHeight,
    offsetY,
    onScroll: handleScroll,
  };
};

