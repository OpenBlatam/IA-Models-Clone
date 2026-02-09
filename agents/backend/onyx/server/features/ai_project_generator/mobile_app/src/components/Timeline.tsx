import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';

interface TimelineItem {
  id: string;
  title: string;
  description?: string;
  timestamp?: string;
  icon?: React.ReactNode;
  status?: 'completed' | 'active' | 'pending';
}

interface TimelineProps {
  items: TimelineItem[];
  orientation?: 'vertical' | 'horizontal';
}

export const Timeline: React.FC<TimelineProps> = ({
  items,
  orientation = 'vertical',
}) => {
  const { theme } = useTheme();

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'completed':
        return theme.success;
      case 'active':
        return theme.primary;
      case 'pending':
        return theme.border;
      default:
        return theme.border;
    }
  };

  if (orientation === 'horizontal') {
    return (
      <View style={styles.horizontalContainer}>
        {items.map((item, index) => (
          <View key={item.id} style={styles.horizontalItem}>
            <View
              style={[
                styles.horizontalDot,
                {
                  backgroundColor: getStatusColor(item.status),
                  borderColor: theme.surface,
                },
              ]}
            >
              {item.icon}
            </View>
            {index < items.length - 1 && (
              <View
                style={[
                  styles.horizontalLine,
                  {
                    backgroundColor: theme.border,
                  },
                ]}
              />
            )}
            <View style={styles.horizontalContent}>
              <Text style={[styles.horizontalTitle, { color: theme.text }]} numberOfLines={1}>
                {item.title}
              </Text>
              {item.timestamp && (
                <Text style={[styles.horizontalTimestamp, { color: theme.textSecondary }]}>
                  {item.timestamp}
                </Text>
              )}
            </View>
          </View>
        ))}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {items.map((item, index) => (
        <View key={item.id} style={styles.item}>
          <View style={styles.leftColumn}>
            <View
              style={[
                styles.dot,
                {
                  backgroundColor: getStatusColor(item.status),
                  borderColor: theme.surface,
                },
              ]}
            >
              {item.icon}
            </View>
            {index < items.length - 1 && (
              <View
                style={[
                  styles.line,
                  {
                    backgroundColor: theme.border,
                  },
                ]}
              />
            )}
          </View>
          <View style={styles.rightColumn}>
            <Text style={[styles.title, { color: theme.text }]}>{item.title}</Text>
            {item.description && (
              <Text style={[styles.description, { color: theme.textSecondary }]}>
                {item.description}
              </Text>
            )}
            {item.timestamp && (
              <Text style={[styles.timestamp, { color: theme.textTertiary }]}>
                {item.timestamp}
              </Text>
            )}
          </View>
        </View>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: spacing.md,
  },
  item: {
    flexDirection: 'row',
    marginBottom: spacing.lg,
  },
  leftColumn: {
    width: 30,
    alignItems: 'center',
    marginRight: spacing.md,
  },
  dot: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 3,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1,
  },
  line: {
    width: 2,
    flex: 1,
    marginTop: spacing.xs,
  },
  rightColumn: {
    flex: 1,
    paddingTop: spacing.xs,
  },
  title: {
    ...typography.body,
    fontWeight: '600',
    marginBottom: spacing.xs,
  },
  description: {
    ...typography.bodySmall,
    marginBottom: spacing.xs,
  },
  timestamp: {
    ...typography.caption,
  },
  horizontalContainer: {
    flexDirection: 'row',
    padding: spacing.md,
  },
  horizontalItem: {
    flex: 1,
    alignItems: 'center',
    position: 'relative',
  },
  horizontalDot: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  horizontalLine: {
    position: 'absolute',
    top: 10,
    left: '50%',
    width: '100%',
    height: 2,
    zIndex: 0,
  },
  horizontalContent: {
    alignItems: 'center',
    marginTop: spacing.sm,
  },
  horizontalTitle: {
    ...typography.caption,
    fontWeight: '600',
    textAlign: 'center',
  },
  horizontalTimestamp: {
    ...typography.caption,
    fontSize: 10,
    marginTop: spacing.xs,
  },
});

