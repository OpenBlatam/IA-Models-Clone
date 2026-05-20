'use client';

import { ReactNode, useRef } from 'react';
import { useVirtualList } from '@/hooks';
import { cn } from '@/utils/classNames';

interface VirtualListProps<T> {
  items: T[];
  itemHeight: number;
  containerHeight: number;
  renderItem: (item: T, index: number) => ReactNode;
  overscan?: number;
  className?: string;
  containerClassName?: string;
}

export function VirtualList<T>({
  items,
  itemHeight,
  containerHeight,
  renderItem,
  overscan = 3,
  className,
  containerClassName,
}: VirtualListProps<T>) {
  const { containerRef, visibleItems, totalHeight, offsetY } = useVirtualList({
    items,
    itemHeight,
    containerHeight,
    overscan,
  });

  return (
    <div
      ref={containerRef}
      className={cn('overflow-auto', containerClassName)}
      style={{ height: containerHeight }}
    >
      <div
        className={cn('relative', className)}
        style={{ height: totalHeight }}
      >
        <div style={{ transform: `translateY(${offsetY}px)` }}>
          {visibleItems.map(({ item, index }) => (
            <div
              key={index}
              style={{ height: itemHeight }}
            >
              {renderItem(item, index)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

