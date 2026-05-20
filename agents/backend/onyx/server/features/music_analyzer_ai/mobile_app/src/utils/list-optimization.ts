import type { ListRenderItem } from 'react-native';
import { Platform } from 'react-native';

/**
 * Creates an optimized renderItem function that prevents unnecessary re-renders
 */
export function createOptimizedRenderItem<T>(
  renderItem: ListRenderItem<T>
): ListRenderItem<T> {
  return (info) => renderItem(info);
}

/**
 * Calculates optimal FlatList props based on item count and screen size
 */
export interface OptimizedListConfig {
  itemHeight: number;
  screenHeight: number;
  overscan?: number;
}

export function getOptimizedListProps(config: OptimizedListConfig) {
  const { itemHeight, screenHeight, overscan = 5 } = config;
  const visibleItems = Math.ceil(screenHeight / itemHeight);

  return {
    initialNumToRender: Math.max(overscan, visibleItems),
    maxToRenderPerBatch: overscan * 2,
    windowSize: overscan * 2,
    removeClippedSubviews: Platform.OS === 'android',
    updateCellsBatchingPeriod: 50,
  };
}

/**
 * Creates a stable key extractor function
 */
export function createKeyExtractor<T>(
  keyPath: keyof T | ((item: T) => string | number)
): (item: T, index: number) => string {
  return (item: T, index: number) => {
    if (typeof keyPath === 'function') {
      return String(keyPath(item));
    }
    return String(item[keyPath] ?? index);
  };
}

/**
 * Calculates optimal item height based on content
 */
export function calculateItemHeight(
  baseHeight: number,
  hasImage: boolean,
  hasSubtitle: boolean,
  hasFooter: boolean
): number {
  let height = baseHeight;
  if (hasImage) height += 60;
  if (hasSubtitle) height += 20;
  if (hasFooter) height += 16;
  return height;
}
