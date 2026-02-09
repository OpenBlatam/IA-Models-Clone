import React from 'react';
import { View, StyleSheet, FlatList, ListRenderItem } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface SwipeableItemProps<T> {
  item: T;
  index: number;
  renderItem: (item: T) => React.ReactNode;
  leftActions?: (item: T) => React.ReactNode;
  rightActions?: (item: T) => React.ReactNode;
  onSwipeLeft?: (item: T) => void;
  onSwipeRight?: (item: T) => void;
}

function SwipeableItem<T>({
  item,
  index,
  renderItem,
  leftActions,
  rightActions,
  onSwipeLeft,
  onSwipeRight,
}: SwipeableItemProps<T>) {
  const { colors } = useTheme();
  const translateX = useSharedValue(0);
  const [isOpen, setIsOpen] = React.useState(false);

  const handleSwipe = (direction: 'left' | 'right') => {
    if (direction === 'left' && onSwipeLeft) {
      onSwipeLeft(item);
    } else if (direction === 'right' && onSwipeRight) {
      onSwipeRight(item);
    }
    translateX.value = withSpring(0);
    runOnJS(setIsOpen)(false);
  };

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [{ translateX: translateX.value }],
    };
  });

  const leftActionsStyle = useAnimatedStyle(() => {
    return {
      opacity: translateX.value > 0 ? 1 : 0,
      transform: [{ translateX: Math.min(translateX.value, 0) }],
    };
  });

  const rightActionsStyle = useAnimatedStyle(() => {
    return {
      opacity: translateX.value < 0 ? 1 : 0,
      transform: [{ translateX: Math.max(translateX.value, 0) }],
    };
  });

  return (
    <View style={styles.container}>
      {/* Left Actions */}
      {leftActions && (
        <Animated.View
          style={[
            styles.actions,
            styles.leftActions,
            { backgroundColor: colors.primary },
            leftActionsStyle,
          ]}
        >
          {leftActions(item)}
        </Animated.View>
      )}

      {/* Right Actions */}
      {rightActions && (
        <Animated.View
          style={[
            styles.actions,
            styles.rightActions,
            { backgroundColor: colors.error },
            rightActionsStyle,
          ]}
        >
          {rightActions(item)}
        </Animated.View>
      )}

      {/* Main Content */}
      <Animated.View
        style={[
          styles.content,
          { backgroundColor: colors.card },
          animatedStyle,
        ]}
      >
        {renderItem(item)}
      </Animated.View>
    </View>
  );
}

interface SwipeableListProps<T> {
  data: T[];
  renderItem: (item: T) => React.ReactNode;
  leftActions?: (item: T) => React.ReactNode;
  rightActions?: (item: T) => React.ReactNode;
  onSwipeLeft?: (item: T) => void;
  onSwipeRight?: (item: T) => void;
  keyExtractor?: (item: T, index: number) => string;
}

function SwipeableList<T>({
  data,
  renderItem,
  leftActions,
  rightActions,
  onSwipeLeft,
  onSwipeRight,
  keyExtractor,
}: SwipeableListProps<T>) {
  const renderItemWithSwipe: ListRenderItem<T> = ({ item, index }) => (
    <SwipeableItem
      item={item}
      index={index}
      renderItem={renderItem}
      leftActions={leftActions}
      rightActions={rightActions}
      onSwipeLeft={onSwipeLeft}
      onSwipeRight={onSwipeRight}
    />
  );

  return (
    <FlatList
      data={data}
      renderItem={renderItemWithSwipe}
      keyExtractor={keyExtractor || ((_, index) => String(index))}
      ItemSeparatorComponent={() => <View style={{ height: 8 }} />}
    />
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    overflow: 'hidden',
  },
  actions: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1,
  },
  leftActions: {
    left: 0,
    paddingLeft: 16,
  },
  rightActions: {
    right: 0,
    paddingRight: 16,
  },
  content: {
    zIndex: 2,
  },
});

export default SwipeableList;

