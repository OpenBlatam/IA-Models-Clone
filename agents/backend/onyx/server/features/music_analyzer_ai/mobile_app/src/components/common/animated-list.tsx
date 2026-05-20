import React, { memo, useRef, useEffect } from 'react';
import { Animated, ListRenderItem } from 'react-native';
import { OptimizedFlatList } from './optimized-flatlist';

interface AnimatedListProps<T> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  animationType?: 'fade' | 'slide' | 'scale';
  animationDuration?: number;
  staggerDelay?: number;
  itemHeight?: number;
  estimatedItemHeight?: number;
}

function AnimatedListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  animationType = 'fade',
  animationDuration = 300,
  staggerDelay = 50,
  itemHeight,
  estimatedItemHeight,
}: AnimatedListProps<T>) {
  const animatedValues = useRef<Map<string, Animated.Value>>(new Map());

  useEffect(() => {
    data.forEach((item) => {
      const key = keyExtractor(item, data.indexOf(item));
      if (!animatedValues.current.has(key)) {
        const animatedValue = new Animated.Value(0);
        animatedValues.current.set(key, animatedValue);

        Animated.timing(animatedValue, {
          toValue: 1,
          duration: animationDuration,
          useNativeDriver: true,
        }).start();
      }
    });
  }, [data, keyExtractor, animationDuration]);

  const animatedRenderItem: ListRenderItem<T> = (info) => {
    const key = keyExtractor(info.item, info.index);
    const animatedValue = animatedValues.current.get(key) || new Animated.Value(1);

    let animatedStyle = {};

    switch (animationType) {
      case 'fade':
        animatedStyle = {
          opacity: animatedValue,
        };
        break;
      case 'slide':
        animatedStyle = {
          opacity: animatedValue,
          transform: [
            {
              translateY: animatedValue.interpolate({
                inputRange: [0, 1],
                outputRange: [20, 0],
              }),
            },
          ],
        };
        break;
      case 'scale':
        animatedStyle = {
          opacity: animatedValue,
          transform: [
            {
              scale: animatedValue.interpolate({
                inputRange: [0, 1],
                outputRange: [0.9, 1],
              }),
            },
          ],
        };
        break;
    }

    return (
      <Animated.View style={animatedStyle}>
        {renderItem(info)}
      </Animated.View>
    );
  };

  return (
    <OptimizedFlatList
      data={data}
      renderItem={animatedRenderItem}
      keyExtractor={keyExtractor}
      itemHeight={itemHeight}
      estimatedItemHeight={estimatedItemHeight}
    />
  );
}

export const AnimatedList = memo(AnimatedListComponent) as <T>(
  props: AnimatedListProps<T>
) => React.ReactElement;

