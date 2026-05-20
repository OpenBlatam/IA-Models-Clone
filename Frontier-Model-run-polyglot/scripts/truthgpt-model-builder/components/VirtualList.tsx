/**
 * Componente VirtualList
 * ======================
 * 
 * Componente de lista virtualizada
 */

'use client'

import React from 'react'
import { useVirtualList, VirtualItem } from '@/lib/hooks/useVirtualList'

export interface VirtualListProps<T> {
  items: T[]
  itemSize: number | ((index: number) => number)
  renderItem: (item: T, index: number, virtualItem: VirtualItem) => React.ReactNode
  overscan?: number
  className?: string
  containerClassName?: string
  itemClassName?: string
}

export default function VirtualList<T>({
  items,
  itemSize,
  renderItem,
  overscan = 5,
  className = '',
  containerClassName = '',
  itemClassName = ''
}: VirtualListProps<T>) {
  const {
    virtualItems,
    totalSize,
    containerRef
  } = useVirtualList({
    itemCount: items.length,
    itemSize,
    overscan
  })

  return (
    <div className={className}>
      <div
        ref={containerRef}
        className={`overflow-auto ${containerClassName}`}
        style={{ height: '100%' }}
      >
        <div style={{ height: `${totalSize}px`, position: 'relative' }}>
          {virtualItems.map((virtualItem) => {
            const item = items[virtualItem.index]
            return (
              <div
                key={virtualItem.index}
                className={itemClassName}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: `${virtualItem.size}px`,
                  transform: `translateY(${virtualItem.start}px)`
                }}
              >
                {renderItem(item, virtualItem.index, virtualItem)}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}






