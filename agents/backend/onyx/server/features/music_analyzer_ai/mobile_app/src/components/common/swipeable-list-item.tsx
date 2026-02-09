import React, { memo, useCallback } from 'react';
import { View, StyleSheet, Animated, PanResponder } from 'react-native';
import { COLORS, SPACING } from '../../constants/config';

interface SwipeableListItemProps {
  children: React.ReactNode;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  leftAction?: React.ReactNode;
  rightAction?: React.ReactNode;
  swipeThreshold?: number;
  disabled?: boolean;
}

function SwipeableListItemComponent({
  children,
  onSwipeLeft,
  onSwipeRight,
  leftAction,
  rightAction,
  swipeThreshold = 100,
  disabled = false,
}: SwipeableListItemProps) {
  const translateX = React.useRef(new Animated.Value(0)).current;
  const panResponder = React.useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => !disabled,
      onMoveShouldSetPanResponder: (_, gestureState) => {
        return !disabled && Math.abs(gestureState.dx) > 10;
      },
      onPanResponderMove: (_, gestureState) => {
        if (!disabled) {
          translateX.setValue(gestureState.dx);
        }
      },
      onPanResponderRelease: (_, gestureState) => {
        if (disabled) return;

        const { dx, vx } = gestureState;
        const shouldSwipeLeft = dx < -swipeThreshold || (dx < 0 && vx < -0.5);
        const shouldSwipeRight = dx > swipeThreshold || (dx > 0 && vx > 0.5);

        if (shouldSwipeLeft && onSwipeLeft) {
          Animated.spring(translateX, {
            toValue: -200,
            useNativeDriver: true,
          }).start(() => {
            onSwipeLeft();
            translateX.setValue(0);
          });
        } else if (shouldSwipeRight && onSwipeRight) {
          Animated.spring(translateX, {
            toValue: 200,
            useNativeDriver: true,
          }).start(() => {
            onSwipeRight();
            translateX.setValue(0);
          });
        } else {
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
          }).start();
        }
      },
    })
  ).current;

  return (
    <View style={styles.container}>
      <View style={styles.actionsContainer}>
        {leftAction && (
          <View style={[styles.leftAction, styles.action]}>
            {leftAction}
          </View>
        )}
        {rightAction && (
          <View style={[styles.rightAction, styles.action]}>
            {rightAction}
          </View>
        )}
      </View>
      <Animated.View
        style={[
          styles.content,
          {
            transform: [{ translateX }],
          },
        ]}
        {...panResponder.panHandlers}
      >
        {children}
      </Animated.View>
    </View>
  );
}

export const SwipeableListItem = memo(SwipeableListItemComponent);

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    overflow: 'hidden',
  },
  actionsContainer: {
    ...StyleSheet.absoluteFillObject,
    flexDirection: 'row',
  },
  leftAction: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'flex-start',
    paddingLeft: SPACING.md,
    backgroundColor: COLORS.error,
  },
  rightAction: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'flex-end',
    paddingRight: SPACING.md,
    backgroundColor: COLORS.success,
  },
  action: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 200,
  },
  content: {
    backgroundColor: COLORS.surface,
    zIndex: 1,
  },
});

