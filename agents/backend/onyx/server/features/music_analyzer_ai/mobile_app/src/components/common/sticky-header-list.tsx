import React, { memo, useState, useCallback, useRef } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { OptimizedFlatList } from './optimized-flatlist';
import { COLORS, SPACING } from '../../constants/config';
import type { ListRenderItem, NativeScrollEvent, NativeSyntheticEvent } from 'react-native';

interface StickyHeaderListProps<T> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  headerComponent: React.ReactNode;
  stickyHeaderComponent?: React.ReactNode;
  itemHeight?: number;
  estimatedItemHeight?: number;
  headerHeight?: number;
  stickyHeaderHeight?: number;
}

function StickyHeaderListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  headerComponent,
  stickyHeaderComponent,
  itemHeight,
  estimatedItemHeight,
  headerHeight = 200,
  stickyHeaderHeight = 60,
}: StickyHeaderListProps<T>) {
  const [showStickyHeader, setShowStickyHeader] = useState(false);
  const scrollY = useRef(new Animated.Value(0)).current;

  const handleScroll = useCallback(
    (event: NativeSyntheticEvent<NativeScrollEvent>) => {
      const offsetY = event.nativeEvent.contentOffset.y;
      const shouldShow = offsetY > headerHeight - stickyHeaderHeight;
      
      if (shouldShow !== showStickyHeader) {
        setShowStickyHeader(shouldShow);
      }

      scrollY.setValue(offsetY);
    },
    [headerHeight, stickyHeaderHeight, showStickyHeader, scrollY]
  );

  const headerOpacity = scrollY.interpolate({
    inputRange: [0, headerHeight - stickyHeaderHeight, headerHeight],
    outputRange: [1, 1, 0],
    extrapolate: 'clamp',
  });

  const stickyHeaderOpacity = scrollY.interpolate({
    inputRange: [0, headerHeight - stickyHeaderHeight, headerHeight],
    outputRange: [0, 0, 1],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.container}>
      {stickyHeaderComponent && (
        <Animated.View
          style={[
            styles.stickyHeader,
            {
              opacity: stickyHeaderOpacity,
              height: stickyHeaderHeight,
            },
          ]}
          pointerEvents={showStickyHeader ? 'auto' : 'none'}
        >
          {stickyHeaderComponent}
        </Animated.View>
      )}
      <OptimizedFlatList
        data={data}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
        itemHeight={itemHeight}
        estimatedItemHeight={estimatedItemHeight}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        ListHeaderComponent={
          <Animated.View style={{ opacity: headerOpacity }}>
            {headerComponent}
          </Animated.View>
        }
      />
    </View>
  );
}

export const StickyHeaderList = memo(StickyHeaderListComponent) as <T>(
  props: StickyHeaderListProps<T>
) => React.ReactElement;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  stickyHeader: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    backgroundColor: COLORS.surface,
    zIndex: 1000,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
    padding: SPACING.md,
  },
});

