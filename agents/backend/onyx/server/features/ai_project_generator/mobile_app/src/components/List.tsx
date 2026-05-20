import React from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { Divider } from './Divider';
import { spacing, borderRadius, typography } from '../theme/colors';

interface ListItem {
  id: string;
  title: string;
  subtitle?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  onPress?: () => void;
}

interface ListProps {
  items: ListItem[];
  showDividers?: boolean;
  dense?: boolean;
}

export const List: React.FC<ListProps> = ({
  items,
  showDividers = true,
  dense = false,
}) => {
  const { theme } = useTheme();

  const renderItem = ({ item, index }: { item: ListItem; index: number }) => (
    <View>
      <TouchableOpacity
        style={[
          styles.item,
          {
            backgroundColor: theme.surface,
            paddingVertical: dense ? spacing.sm : spacing.md,
          },
        ]}
        onPress={item.onPress}
        activeOpacity={item.onPress ? 0.7 : 1}
        disabled={!item.onPress}
      >
        {item.leftIcon && (
          <View style={styles.leftIconContainer}>{item.leftIcon}</View>
        )}
        <View style={styles.content}>
          <Text style={[styles.title, { color: theme.text }]} numberOfLines={1}>
            {item.title}
          </Text>
          {item.subtitle && (
            <Text style={[styles.subtitle, { color: theme.textSecondary }]} numberOfLines={2}>
              {item.subtitle}
            </Text>
          )}
        </View>
        {item.rightIcon && (
          <View style={styles.rightIconContainer}>{item.rightIcon}</View>
        )}
      </TouchableOpacity>
      {showDividers && index < items.length - 1 && (
        <Divider spacing={0} />
      )}
    </View>
  );

  return (
    <View style={[styles.container, { backgroundColor: theme.surface }]}>
      <FlatList
        data={items}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        scrollEnabled={false}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
  },
  leftIconContainer: {
    marginRight: spacing.md,
  },
  content: {
    flex: 1,
  },
  title: {
    ...typography.body,
    fontWeight: '500',
  },
  subtitle: {
    ...typography.bodySmall,
    marginTop: spacing.xs,
  },
  rightIconContainer: {
    marginLeft: spacing.md,
  },
});

