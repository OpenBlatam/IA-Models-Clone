import React, { memo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ProjectStatus } from '../types';
import { colors, spacing, borderRadius, typography } from '../theme/colors';

interface StatusBadgeProps {
  status: ProjectStatus | string;
}

const getStatusColor = (status: ProjectStatus | string): string => {
  switch (status) {
    case ProjectStatus.COMPLETED:
      return colors.status.completed;
    case ProjectStatus.PROCESSING:
      return colors.status.processing;
    case ProjectStatus.QUEUED:
      return colors.status.queued;
    case ProjectStatus.FAILED:
      return colors.status.failed;
    case ProjectStatus.CANCELLED:
      return colors.status.cancelled;
    default:
      return colors.textTertiary;
  }
};

export const StatusBadge: React.FC<StatusBadgeProps> = memo(({ status }) => {
  const color = getStatusColor(status);

  return (
    <View style={[styles.badge, { backgroundColor: color }]}>
      <Text style={styles.text}>{status.toUpperCase()}</Text>
    </View>
  );
});

StatusBadge.displayName = 'StatusBadge';

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
    alignSelf: 'flex-start',
  },
  text: {
    ...typography.caption,
    color: colors.surface,
    fontWeight: '600',
  },
});
