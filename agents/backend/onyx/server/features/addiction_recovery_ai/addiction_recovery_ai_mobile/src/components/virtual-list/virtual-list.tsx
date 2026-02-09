import React, { memo, useMemo, useState, useCallback } from 'react';
import { View, ScrollView, ScrollViewProps } from 'react-native';
import { calculateVisibleItems, type VirtualListConfig } from '@/utils/virtual-list';

interface VirtualListProps<T> extends Omit<ScrollViewProps, 'children'> {
  data: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
}

function VirtualListComponent<T>({
  data,
  renderItem,
  itemHeight,
  containerHeight,
  overscan = 2,
  ...scrollViewProps
}: VirtualListProps<T>): JSX.Element {
  const [scrollOffset, setScrollOffset] = useState(0);

  const config: VirtualListConfig = useMemo(
    () => ({
      itemHeight,
      containerHeight,
      overscan,
    }),
    [itemHeight, containerHeight, overscan]
  );

  const visibleItems = useMemo(
    () => calculateVisibleItems(scrollOffset, config),
    [scrollOffset, config]
  );

  const totalHeight = useMemo(
    () => data.length * itemHeight,
    [data.length, itemHeight]
  );

  const handleScroll = useCallback(
    (event: { nativeEvent: { contentOffset: { y: number } } }) => {
      setScrollOffset(event.nativeEvent.contentOffset.y);
    },
    []
  );

  return (
    <ScrollView
      {...scrollViewProps}
      onScroll={handleScroll}
      scrollEventThrottle={16}
      contentContainerStyle={[
        { height: totalHeight },
        scrollViewProps.contentContainerStyle,
      ]}
    >
      {visibleItems.map(({ index, start }) => {
        if (index >= data.length) {
          return null;
        }

        return (
          <View
            key={index}
            style={{
              position: 'absolute',
              top: start,
              left: 0,
              right: 0,
              height: itemHeight,
            }}
          >
            {renderItem(data[index], index)}
          </View>
        );
      })}
    </ScrollView>
  );
}

export const VirtualList = memo(VirtualListComponent) as <T>(
  props: VirtualListProps<T>
) => JSX.Element;

