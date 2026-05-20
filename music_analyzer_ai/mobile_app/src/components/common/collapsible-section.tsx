import React, { memo, useState, useCallback, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useHapticFeedback } from '../../hooks/use-haptic-feedback';

interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  onToggle?: (expanded: boolean) => void;
  headerComponent?: React.ReactNode;
  icon?: string;
  animated?: boolean;
}

function CollapsibleSectionComponent({
  title,
  children,
  defaultExpanded = false,
  onToggle,
  headerComponent,
  icon,
  animated = true,
}: CollapsibleSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const haptics = useHapticFeedback();
  const animation = useRef(new Animated.Value(defaultExpanded ? 1 : 0)).current;
  const contentHeight = useRef<number>(0);

  const toggle = useCallback(() => {
    haptics.light();
    const newExpanded = !isExpanded;
    setIsExpanded(newExpanded);
    onToggle?.(newExpanded);

    if (animated) {
      Animated.spring(animation, {
        toValue: newExpanded ? 1 : 0,
        useNativeDriver: false,
        tension: 100,
        friction: 8,
      }).start();
    }
  }, [isExpanded, onToggle, animated, animation, haptics]);

  const rotateInterpolate = animation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '180deg'],
  });

  const heightInterpolate = animation.interpolate({
    inputRange: [0, 1],
    outputRange: [0, contentHeight.current || 1000],
  });

  const opacityInterpolate = animation.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [0, 0, 1],
  });

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.header}
        onPress={toggle}
        activeOpacity={0.7}
        accessibilityRole="button"
        accessibilityState={{ expanded: isExpanded }}
      >
        {headerComponent || (
          <View style={styles.headerContent}>
            {icon && <Text style={styles.icon}>{icon}</Text>}
            <Text style={styles.title}>{title}</Text>
          </View>
        )}
        <Animated.View
          style={[
            styles.chevron,
            {
              transform: [{ rotate: rotateInterpolate }],
            },
          ]}
        >
          <Text style={styles.chevronText}>▼</Text>
        </Animated.View>
      </TouchableOpacity>

      {animated ? (
        <Animated.View
          style={[
            styles.content,
            {
              maxHeight: heightInterpolate,
              opacity: opacityInterpolate,
            },
          ]}
          onLayout={(event) => {
            contentHeight.current = event.nativeEvent.layout.height;
          }}
        >
          {children}
        </Animated.View>
      ) : (
        isExpanded && <View style={styles.content}>{children}</View>
      )}
    </View>
  );
}

export const CollapsibleSection = memo(CollapsibleSectionComponent);

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.md,
    marginBottom: SPACING.sm,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: SPACING.md,
    backgroundColor: COLORS.surface,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  icon: {
    fontSize: 20,
    marginRight: SPACING.sm,
  },
  title: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    flex: 1,
  },
  chevron: {
    marginLeft: SPACING.sm,
  },
  chevronText: {
    fontSize: 12,
    color: COLORS.textSecondary,
  },
  content: {
    padding: SPACING.md,
    paddingTop: 0,
  },
});

