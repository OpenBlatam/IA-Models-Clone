/**
 * Virtualization utilities for large lists
 * @module robot-3d-view/utils/virtualization
 */

/**
 * Virtual item
 */
export interface VirtualItem {
  index: number;
  start: number;
  end: number;
  size: number;
}

/**
 * Virtualization options
 */
export interface VirtualizationOptions {
  itemHeight: number | ((index: number) => number);
  containerHeight: number;
  overscan?: number;
}

/**
 * Virtualization result
 */
export interface VirtualizationResult {
  items: VirtualItem[];
  totalHeight: number;
  startOffset: number;
  endOffset: number;
}

/**
 * Calculates virtual items for a list
 * 
 * @param count - Total number of items
 * @param scrollTop - Current scroll position
 * @param options - Virtualization options
 * @returns Virtualization result
 */
export function virtualize(
  count: number,
  scrollTop: number,
  options: VirtualizationOptions
): VirtualizationResult {
  const { itemHeight, containerHeight, overscan = 3 } = options;

  const getItemHeight = (index: number): number => {
    return typeof itemHeight === 'function' ? itemHeight(index) : itemHeight;
  };

  // Calculate total height
  let totalHeight = 0;
  const heights: number[] = [];
  for (let i = 0; i < count; i++) {
    const height = getItemHeight(i);
    heights.push(height);
    totalHeight += height;
  }

  // Find start index
  let startIndex = 0;
  let accumulatedHeight = 0;
  for (let i = 0; i < count; i++) {
    if (accumulatedHeight + heights[i] > scrollTop) {
      startIndex = Math.max(0, i - overscan);
      break;
    }
    accumulatedHeight += heights[i];
  }

  // Find end index
  let endIndex = count;
  let visibleHeight = 0;
  for (let i = startIndex; i < count; i++) {
    visibleHeight += heights[i];
    if (visibleHeight > containerHeight + scrollTop) {
      endIndex = Math.min(count, i + overscan + 1);
      break;
    }
  }

  // Calculate offsets
  let startOffset = 0;
  for (let i = 0; i < startIndex; i++) {
    startOffset += heights[i];
  }

  let endOffset = 0;
  for (let i = endIndex; i < count; i++) {
    endOffset += heights[i];
  }

  // Create virtual items
  const items: VirtualItem[] = [];
  let currentOffset = startOffset;
  for (let i = startIndex; i < endIndex; i++) {
    const size = heights[i];
    items.push({
      index: i,
      start: currentOffset,
      end: currentOffset + size,
      size,
    });
    currentOffset += size;
  }

  return {
    items,
    totalHeight,
    startOffset,
    endOffset,
  };
}

/**
 * Hook-like function for virtualization
 */
export function useVirtualization(
  count: number,
  scrollTop: number,
  options: VirtualizationOptions
): VirtualizationResult {
  return virtualize(count, scrollTop, options);
}



