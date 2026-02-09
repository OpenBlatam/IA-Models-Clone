import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface CollapsibleProps {
  title: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  icon?: React.ReactNode;
  onToggle?: (expanded: boolean) => void;
}

export const Collapsible: React.FC<CollapsibleProps> = ({
  title,
  children,
  defaultExpanded = false,
  icon,
  onToggle,
}) => {
  const { theme } = useTheme();
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [animation] = useState(new Animated.Value(defaultExpanded ? 1 : 0));

  const toggle = () => {
    const newExpanded = !expanded;
    setExpanded(newExpanded);
    hapticFeedback.selection();

    Animated.timing(animation, {
      toValue: newExpanded ? 1 : 0,
      duration: 300,
      useNativeDriver: false,
    }).start();

    onToggle?.(newExpanded);
  };

  const rotate = animation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '180deg'],
  });

  const maxHeight = animation.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 1000],
  });

  const opacity = animation.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [0, 0.5, 1],
  });

  return (
    <View style={[styles.container, { backgroundColor: theme.surface, borderColor: theme.border }]}>
      <TouchableOpacity
        style={styles.header}
        onPress={toggle}
        activeOpacity={0.7}
      >
        <View style={styles.headerContent}>
          {icon && <View style={styles.iconContainer}>{icon}</View>}
          <Text style={[styles.title, { color: theme.text }]}>{title}</Text>
        </View>
        <Animated.View style={{ transform: [{ rotate }] }}>
          <Text style={[styles.arrow, { color: theme.textSecondary }]}>▼</Text>
        </Animated.View>
      </TouchableOpacity>
      <Animated.View
        style={[
          styles.content,
          {
            maxHeight,
            opacity,
          },
        ]}
      >
        <View style={styles.contentInner}>{children}</View>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderWidth: 1,
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    marginBottom: spacing.md,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.md,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: spacing.sm,
  },
  iconContainer: {
    marginRight: spacing.xs,
  },
  title: {
    ...typography.body,
    fontWeight: '600',
    flex: 1,
  },
  arrow: {
    fontSize: 12,
    marginLeft: spacing.sm,
  },
  content: {
    overflow: 'hidden',
  },
  contentInner: {
    padding: spacing.md,
    paddingTop: 0,
  },
});

