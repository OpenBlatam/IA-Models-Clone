export interface VirtualListConfig {
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
}

export interface VirtualListItem {
  index: number;
  start: number;
  end: number;
}

export function calculateVisibleItems(
  scrollOffset: number,
  config: VirtualListConfig
): VirtualListItem[] {
  const { itemHeight, containerHeight, overscan = 2 } = config;
  
  const startIndex = Math.max(0, Math.floor(scrollOffset / itemHeight) - overscan);
  const endIndex = Math.min(
    Math.ceil((scrollOffset + containerHeight) / itemHeight) + overscan,
    Infinity
  );

  const items: VirtualListItem[] = [];
  for (let i = startIndex; i < endIndex; i++) {
    items.push({
      index: i,
      start: i * itemHeight,
      end: (i + 1) * itemHeight,
    });
  }

  return items;
}

