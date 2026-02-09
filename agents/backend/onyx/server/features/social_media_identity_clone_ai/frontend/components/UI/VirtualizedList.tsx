import { memo, useMemo } from 'react';
import { cn } from '@/lib/utils';

interface VirtualizedListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  itemHeight?: number;
  containerHeight?: number;
  className?: string;
  emptyMessage?: string;
}

const VirtualizedList = <T,>({
  items,
  renderItem,
  itemHeight = 100,
  containerHeight = 400,
  className = '',
  emptyMessage = 'No items found',
}: VirtualizedListProps<T>): JSX.Element => {
  const visibleItems = useMemo(() => {
    if (items.length === 0) {
      return [];
    }
    
    const visibleCount = Math.ceil(containerHeight / itemHeight) + 2;
    return items.slice(0, visibleCount);
  }, [items, containerHeight, itemHeight]);

  if (items.length === 0) {
    return (
      <div className={cn('flex items-center justify-center h-full', className)}>
        <p className="text-gray-500">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div
      className={cn('overflow-auto', className)}
      style={{ height: containerHeight }}
      role="list"
    >
      {visibleItems.map((item, index) => (
        <div key={index} role="listitem" style={{ minHeight: itemHeight }}>
          {renderItem(item, index)}
        </div>
      ))}
    </div>
  );
};

export default memo(VirtualizedList) as typeof VirtualizedList;



