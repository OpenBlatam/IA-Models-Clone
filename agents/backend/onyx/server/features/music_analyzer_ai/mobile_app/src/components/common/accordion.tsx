import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface AccordionItem {
  title: string;
  content: React.ReactNode;
  key: string;
}

interface AccordionProps {
  items: AccordionItem[];
  allowMultiple?: boolean;
  defaultOpen?: string[];
}

/**
 * Accordion component
 * Collapsible sections
 */
export function Accordion({
  items,
  allowMultiple = false,
  defaultOpen = [],
}: AccordionProps) {
  const [openKeys, setOpenKeys] = useState<string[]>(defaultOpen);

  const toggleItem = (key: string) => {
    setOpenKeys((prev) => {
      if (prev.includes(key)) {
        return prev.filter((k) => k !== key);
      }
      return allowMultiple ? [...prev, key] : [key];
    });
  };

  return (
    <View style={styles.container}>
      {items.map((item) => {
        const isOpen = openKeys.includes(item.key);
        const height = useSharedValue(isOpen ? 1 : 0);

        React.useEffect(() => {
          height.value = withTiming(isOpen ? 1 : 0, { duration: 300 });
        }, [isOpen, height]);

        const animatedStyle = useAnimatedStyle(() => ({
          maxHeight: height.value * 1000,
          opacity: height.value,
        }));

        return (
          <View key={item.key} style={styles.item}>
            <TouchableOpacity
              style={styles.header}
              onPress={() => toggleItem(item.key)}
              accessibilityRole="button"
              accessibilityState={{ expanded: isOpen }}
            >
              <Text style={styles.title}>{item.title}</Text>
              <Text style={styles.icon}>{isOpen ? '−' : '+'}</Text>
            </TouchableOpacity>
            <Animated.View style={[styles.content, animatedStyle]}>
              {item.content}
            </Animated.View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  item: {
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: SPACING.md,
    backgroundColor: COLORS.surface,
  },
  title: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    flex: 1,
  },
  icon: {
    ...TYPOGRAPHY.h2,
    color: COLORS.textSecondary,
    marginLeft: SPACING.sm,
  },
  content: {
    padding: SPACING.md,
    backgroundColor: COLORS.background,
    overflow: 'hidden',
  },
});

