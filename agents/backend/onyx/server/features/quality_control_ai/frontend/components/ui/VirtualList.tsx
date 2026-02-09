'use client';

import { memo, type ReactNode } from 'react';
import { useVirtualList } from '@/lib/hooks';
import { cn } from '@/lib/utils';

interface VirtualListProps<T> {
  items: T[];
  itemHeight: number | ((index: number) => number);
  containerHeight: number;
  renderItem: (item: T, index: number) => ReactNode;
  className?: string;
  overscan?: number;
}

const VirtualList = memo(
  <T,>({
    items,
    itemHeight,
    containerHeight,
    renderItem,
    className,
    overscan = 3,
  }: VirtualListProps<T>): JSX.Element => {
    const { containerRef, visibleItems, totalHeight, offsetY, onScroll } =
      useVirtualList(items, {
        itemHeight,
        containerHeight,
        overscan,
      });

    return (
      <div
        ref={containerRef}
        className={cn('overflow-auto', className)}
        style={{ height: containerHeight }}
        onScroll={onScroll}
      >
        <div style={{ height: totalHeight, position: 'relative' }}>
          <div style={{ transform: `translateY(${offsetY}px)` }}>
            {visibleItems.map(({ item, index }) => (
              <div key={index}>{renderItem(item, index)}</div>
            ))}
          </div>
        </div>
      </div>
    );
  }
) as <T,>(props: VirtualListProps<T>) => JSX.Element;

VirtualList.displayName = 'VirtualList';

export default VirtualList;

