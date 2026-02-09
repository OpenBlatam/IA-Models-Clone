/**
 * Hooks module exports
 * Centralized export point for all custom hooks
 */

// Data Management Hooks
export { usePagination } from './usePagination';
export { useSort } from './useSort';
export { useFilter } from './useFilter';
export { useCombinedData } from './useCombinedData';
export { useInfiniteScroll } from './useInfiniteScroll';
export { useMutation } from './useMutation';

// UI Hooks
export { useModal } from './useModal';
export { useToastManager } from './useToastManager';
export { useConfirm } from './useConfirm';
export { useLocalStorage } from './useLocalStorage';

// Media Hooks
export { useMedia } from './useMedia';
export { useImageOptimization, useImagePreloader } from './useImageOptimization';

// Performance Hooks
export {
  useOptimizedFlatList,
  useOptimizedRenderItem,
  useOptimizedKeyExtractor,
} from './useOptimizedFlatList';
export { usePerformance } from './usePerformance';
export { useMemoizedCallback } from './useMemoizedCallback';

// Responsive Hooks
export { useResponsive, useResponsiveValue } from './useResponsive';

// System Hooks
export { usePermissions } from './usePermissions';
export { useDeepLinking } from './useDeepLinking';
export { useNetworkStatus } from './useNetworkStatus';

