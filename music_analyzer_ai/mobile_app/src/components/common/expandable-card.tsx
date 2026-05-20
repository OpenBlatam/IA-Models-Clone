import React, { memo, useState, useCallback, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated } from 'react-native';
import { Card } from './card';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useHapticFeedback } from '../../hooks/use-haptic-feedback';

interface ExpandableCardProps {
  title: string;
  summary: string;
  expandedContent: React.ReactNode;
  defaultExpanded?: boolean;
  onExpand?: (expanded: boolean) => void;
  icon?: string;
  variant?: 'default' | 'compact';
}

function ExpandableCardComponent({
  title,
  summary,
  expandedContent,
  defaultExpanded = false,
  onExpand,
  icon,
  variant = 'default',
}: ExpandableCardProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const haptics = useHapticFeedback();
  const animation = useRef(new Animated.Value(defaultExpanded ? 1 : 0)).current;
  const contentHeight = useRef<number>(0);

  const toggle = useCallback(() => {
    haptics.light();
    const newExpanded = !isExpanded;
    setIsExpanded(newExpanded);
    onExpand?.(newExpanded);

    Animated.spring(animation, {
      toValue: newExpanded ? 1 : 0,
      useNativeDriver: false,
      tension: 100,
      friction: 8,
    }).start();
  }, [isExpanded, onExpand, animation, haptics]);

  const rotateInterpolate = animation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '180deg'],
  });

  const heightInterpolate = animation.interpolate({
    inputRange: [0, 1],
    outputRange: [0, contentHeight.current || 1000],
  });

  const opacityInterpolate = animation.interpolate({
    inputRange: [0, 0.3, 1],
    outputRange: [0, 0, 1],
  });

  return (
    <Card style={variant === 'compact' ? styles.compactCard : styles.card}>
      <TouchableOpacity
        onPress={toggle}
        activeOpacity={0.7}
        style={styles.header}
        accessibilityRole="button"
        accessibilityState={{ expanded: isExpanded }}
      >
        <View style={styles.headerContent}>
          {icon && <Text style={styles.icon}>{icon}</Text>}
          <View style={styles.titleContainer}>
            <Text style={styles.title}>{title}</Text>
            {!isExpanded && (
              <Text style={styles.summary} numberOfLines={variant === 'compact' ? 1 : 2}>
                {summary}
              </Text>
            )}
          </View>
        </View>
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

      <Animated.View
        style={[
          styles.expandedContent,
          {
            maxHeight: heightInterpolate,
            opacity: opacityInterpolate,
          },
        ]}
        onLayout={(event) => {
          contentHeight.current = event.nativeEvent.layout.height;
        }}
      >
        {expandedContent}
      </Animated.View>
    </Card>
  );
}

export const ExpandableCard = memo(ExpandableCardComponent);

const styles = StyleSheet.create({
  card: {
    marginBottom: SPACING.md,
  },
  compactCard: {
    marginBottom: SPACING.sm,
    padding: SPACING.sm,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  headerContent: {
    flexDirection: 'row',
    flex: 1,
    marginRight: SPACING.sm,
  },
  icon: {
    fontSize: 24,
    marginRight: SPACING.sm,
  },
  titleContainer: {
    flex: 1,
  },
  title: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    marginBottom: SPACING.xs,
  },
  summary: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
  },
  chevron: {
    marginTop: SPACING.xs,
  },
  chevronText: {
    fontSize: 12,
    color: COLORS.textSecondary,
  },
  expandedContent: {
    marginTop: SPACING.md,
    overflow: 'hidden',
  },
});

