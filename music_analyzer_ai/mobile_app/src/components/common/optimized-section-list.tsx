import React, { memo, useMemo, useCallback } from 'react';
import { SectionList, SectionListProps, Platform, SectionListRenderItem } from 'react-native';

interface OptimizedSectionListProps<T, S> extends Omit<SectionListProps<T, S>, 'sections' | 'renderItem' | 'getItemLayout'> {
  sections: SectionListProps<T, S>['sections'];
  renderItem: SectionListRenderItem<T, S>;
  itemHeight?: number;
  sectionHeaderHeight?: number;
  estimatedItemHeight?: number;
  estimatedSectionHeaderHeight?: number;
  overscan?: number;
  enableOptimizations?: boolean;
}

function OptimizedSectionListComponent<T, S>({
  sections,
  renderItem,
  itemHeight,
  sectionHeaderHeight,
  estimatedItemHeight,
  estimatedSectionHeaderHeight,
  overscan = 5,
  enableOptimizations = true,
  ...props
}: OptimizedSectionListProps<T, S>) {
  const memoizedRenderItem = useCallback<SectionListRenderItem<T, S>>(
    (info) => renderItem(info),
    [renderItem]
  );

  const getItemLayout = useMemo(() => {
    if (!itemHeight || !sectionHeaderHeight) {
      return undefined;
    }

    return (
      _data: Array<{ data: T[] }> | null | undefined,
      index: number,
      sectionIndex: number
    ) => {
      let offset = sectionHeaderHeight * (sectionIndex + 1);
      for (let i = 0; i < sectionIndex; i++) {
        const section = sections[i];
        if (section?.data) {
          offset += section.data.length * itemHeight;
        }
      }
      return {
        length: itemHeight,
        offset,
        index,
      };
    };
  }, [itemHeight, sectionHeaderHeight, sections]);

  const optimizationProps = useMemo(() => {
    if (!enableOptimizations) {
      return {};
    }

    return {
      removeClippedSubviews: Platform.OS === 'android',
      maxToRenderPerBatch: overscan * 2,
      windowSize: overscan * 2,
      initialNumToRender: overscan,
      updateCellsBatchingPeriod: 50,
      ...(getItemLayout && { getItemLayout }),
      ...(estimatedItemHeight && !getItemLayout && { estimatedItemSize: estimatedItemHeight }),
      ...(estimatedSectionHeaderHeight && { estimatedSectionHeaderSize: estimatedSectionHeaderHeight }),
    };
  }, [enableOptimizations, overscan, getItemLayout, estimatedItemHeight, estimatedSectionHeaderHeight]);

  return (
    <SectionList
      sections={sections}
      renderItem={memoizedRenderItem}
      {...optimizationProps}
      {...props}
    />
  );
}

export const OptimizedSectionList = memo(OptimizedSectionListComponent) as <T, S>(
  props: OptimizedSectionListProps<T, S>
) => React.ReactElement;
