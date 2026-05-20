import React, { useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  PanResponder,
  TouchableOpacity,
} from 'react-native';
import { colors, spacing, borderRadius, typography } from '../theme/colors';

interface SwipeableCardProps {
  children: React.ReactNode;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  leftAction?: {
    label: string;
    color: string;
    icon?: string;
  };
  rightAction?: {
    label: string;
    color: string;
    icon?: string;
  };
  swipeThreshold?: number;
}

export const SwipeableCard: React.FC<SwipeableCardProps> = ({
  children,
  onSwipeLeft,
  onSwipeRight,
  leftAction,
  rightAction,
  swipeThreshold = 100,
}) => {
  const pan = useRef(new Animated.ValueXY()).current;
  const opacity = useRef(new Animated.Value(1)).current;

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: () => {
        pan.setOffset({
          x: (pan.x as any)._value,
          y: (pan.y as any)._value,
        });
      },
      onPanResponderMove: (_, gestureState) => {
        pan.setValue({ x: gestureState.dx, y: 0 });
        const absDx = Math.abs(gestureState.dx);
        opacity.setValue(Math.max(0, 1 - absDx / 200));
      },
      onPanResponderRelease: (_, gestureState) => {
        pan.flattenOffset();
        const { dx } = gestureState;

        if (Math.abs(dx) > swipeThreshold) {
          if (dx > 0 && onSwipeRight) {
            Animated.parallel([
              Animated.timing(pan, {
                toValue: { x: 500, y: 0 },
                duration: 200,
                useNativeDriver: true,
              }),
              Animated.timing(opacity, {
                toValue: 0,
                duration: 200,
                useNativeDriver: true,
              }),
            ]).start(() => {
              onSwipeRight();
              pan.setValue({ x: 0, y: 0 });
              opacity.setValue(1);
            });
          } else if (dx < 0 && onSwipeLeft) {
            Animated.parallel([
              Animated.timing(pan, {
                toValue: { x: -500, y: 0 },
                duration: 200,
                useNativeDriver: true,
              }),
              Animated.timing(opacity, {
                toValue: 0,
                duration: 200,
                useNativeDriver: true,
              }),
            ]).start(() => {
              onSwipeLeft();
              pan.setValue({ x: 0, y: 0 });
              opacity.setValue(1);
            });
          } else {
            Animated.spring(pan, {
              toValue: { x: 0, y: 0 },
              useNativeDriver: true,
              tension: 50,
              friction: 7,
            }).start();
            Animated.spring(opacity, {
              toValue: 1,
              useNativeDriver: true,
              tension: 50,
              friction: 7,
            }).start();
          }
        } else {
          Animated.spring(pan, {
            toValue: { x: 0, y: 0 },
            useNativeDriver: true,
            tension: 50,
            friction: 7,
          }).start();
          Animated.spring(opacity, {
            toValue: 1,
            useNativeDriver: true,
            tension: 50,
            friction: 7,
          }).start();
        }
      },
    })
  ).current;

  const leftActionOpacity = pan.x.interpolate({
    inputRange: [-swipeThreshold, 0],
    outputRange: [1, 0],
    extrapolate: 'clamp',
  });

  const rightActionOpacity = pan.x.interpolate({
    inputRange: [0, swipeThreshold],
    outputRange: [0, 1],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.container}>
      {leftAction && (
        <Animated.View
          style={[
            styles.action,
            styles.leftAction,
            { backgroundColor: leftAction.color, opacity: leftActionOpacity },
          ]}
        >
          <Text style={styles.actionIcon}>{leftAction.icon || '←'}</Text>
          <Text style={styles.actionLabel}>{leftAction.label}</Text>
        </Animated.View>
      )}
      {rightAction && (
        <Animated.View
          style={[
            styles.action,
            styles.rightAction,
            { backgroundColor: rightAction.color, opacity: rightActionOpacity },
          ]}
        >
          <Text style={styles.actionIcon}>{rightAction.icon || '→'}</Text>
          <Text style={styles.actionLabel}>{rightAction.label}</Text>
        </Animated.View>
      )}
      <Animated.View
        style={[
          {
            transform: [{ translateX: pan.x }],
            opacity,
          },
        ]}
        {...panResponder.panHandlers}
      >
        {children}
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  action: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 100,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 0,
  },
  leftAction: {
    left: 0,
    borderTopLeftRadius: borderRadius.lg,
    borderBottomLeftRadius: borderRadius.lg,
  },
  rightAction: {
    right: 0,
    borderTopRightRadius: borderRadius.lg,
    borderBottomRightRadius: borderRadius.lg,
  },
  actionIcon: {
    fontSize: 24,
    color: colors.surface,
    marginBottom: spacing.xs,
  },
  actionLabel: {
    ...typography.caption,
    color: colors.surface,
    fontWeight: '600',
    textAlign: 'center',
  },
});

