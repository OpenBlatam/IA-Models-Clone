'use client';

import { ReactNode, useMemo } from 'react';
import { FixedSizeList as List, VariableSizeList, ListChildComponentProps } from 'react-window';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface VirtualizedListProps<T> {
  items: T[];
  height: number;
  itemHeight?: number | ((index: number) => number);
  renderItem: (item: T, index: number) => ReactNode;
  className?: string;
  gap?: number;
  overscanCount?: number;
}

export default function VirtualizedList<T>({
  items,
  height,
  itemHeight = 80,
  renderItem,
  className,
  gap = 8,
  overscanCount = 5,
}: VirtualizedListProps<T>) {
  const isVariableHeight = typeof itemHeight === 'function';

  if (isVariableHeight) {
    const getItemSize = itemHeight as (index: number) => number;
    const itemSizes = useMemo(
      () => items.map((_, index) => getItemSize(index)),
      [items, itemHeight]
    );

    return (
      <VariableSizeList
        height={height}
        itemCount={items.length}
        itemSize={(index) => itemSizes[index] + gap}
        width="100%"
        overscanCount={overscanCount}
        className={cn('scrollbar-hide', className)}
      >
        {({ index, style }: ListChildComponentProps) => (
          <div style={{ ...style, paddingBottom: gap }}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2, delay: index * 0.02 }}
            >
              {renderItem(items[index], index)}
            </motion.div>
          </div>
        )}
      </VariableSizeList>
    );
  }

  return (
    <List
      height={height}
      itemCount={items.length}
      itemSize={(itemHeight as number) + gap}
      width="100%"
      overscanCount={overscanCount}
      className={cn('scrollbar-hide', className)}
    >
      {({ index, style }: ListChildComponentProps) => (
        <div style={{ ...style, paddingBottom: gap }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, delay: index * 0.02 }}
          >
            {renderItem(items[index], index)}
          </motion.div>
        </div>
      )}
    </List>
  );
}



