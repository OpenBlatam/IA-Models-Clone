import React, { useMemo, useCallback } from 'react';
import type { FlatListProps } from 'react-native';

interface UseOptimizedFlatListOptions<T> {
  itemHeight?: number;
  estimatedItemSize?: number;
  windowSize?: number;
  maxToRenderPerBatch?: number;
  updateCellsBatchingPeriod?: number;
  initialNumToRender?: number;
}

/**
 * Hook para optimizar FlatList con mejores prácticas de rendimiento
 * @param options - Opciones de optimización
 * @returns Props optimizadas para FlatList
 */
export const useOptimizedFlatList = <T,>(
  options: UseOptimizedFlatListOptions<T> = {}
): Partial<FlatListProps<T>> => {
  const {
    itemHeight,
    estimatedItemSize = 50,
    windowSize = 10,
    maxToRenderPerBatch = 10,
    updateCellsBatchingPeriod = 50,
    initialNumToRender = 10,
  } = options;

  const getItemLayout = useMemo(() => {
    if (itemHeight) {
      return (_: unknown, index: number) => ({
        length: itemHeight,
        offset: itemHeight * index,
        index,
      });
    }
    return undefined;
  }, [itemHeight]);

  const optimizedProps = useMemo<Partial<FlatListProps<T>>>(
    () => ({
      removeClippedSubviews: true,
      maxToRenderPerBatch,
      updateCellsBatchingPeriod,
      windowSize,
      initialNumToRender,
      getItemLayout,
    }),
    [
      maxToRenderPerBatch,
      updateCellsBatchingPeriod,
      windowSize,
      initialNumToRender,
      getItemLayout,
    ]
  );

  return optimizedProps;
};

/**
 * Hook para crear renderItem optimizado con useCallback
 * @param renderItem - Función de renderizado
 * @param deps - Dependencias para useCallback
 * @returns Función renderItem memoizada
 */
export const useOptimizedRenderItem = <T,>(
  renderItem: (item: T, index: number) => React.ReactElement,
  deps: React.DependencyList = []
) => {
  return useCallback(
    ({ item, index }: { item: T; index: number }) => renderItem(item, index),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    deps
  );
};

/**
 * Hook para crear keyExtractor optimizado
 * @param keyExtractor - Función para extraer la clave
 * @param deps - Dependencias para useCallback
 * @returns Función keyExtractor memoizada
 */
export const useOptimizedKeyExtractor = <T,>(
  keyExtractor: (item: T, index: number) => string,
  deps: React.DependencyList = []
) => {
  return useCallback(
    (item: T, index: number) => keyExtractor(item, index),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    deps
  );
};

