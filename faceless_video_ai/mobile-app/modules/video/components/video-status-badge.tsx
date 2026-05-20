import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '@/contexts/theme-context';
import { VIDEO_STATUS_COLORS, VIDEO_STATUS_LABELS } from '@/utils/constants';
import type { VideoStatus } from '@/types/api';

interface VideoStatusBadgeProps {
  status: VideoStatus;
  size?: 'small' | 'medium' | 'large';
}

export function VideoStatusBadge({ status, size = 'medium' }: VideoStatusBadgeProps) {
  const { colors } = useTheme();
  const statusColor = VIDEO_STATUS_COLORS[status] || colors.text;
  const statusLabel = VIDEO_STATUS_LABELS[status] || status;

  const sizeStyles = {
    small: { padding: 4, fontSize: 10 },
    medium: { padding: 6, fontSize: 12 },
    large: { padding: 8, fontSize: 14 },
  };

  return (
    <View
      style={[
        styles.badge,
        {
          backgroundColor: `${statusColor}20`,
          paddingHorizontal: sizeStyles[size].padding,
          paddingVertical: sizeStyles[size].padding / 2,
        },
      ]}
    >
      <Text
        style={[
          styles.text,
          {
            color: statusColor,
            fontSize: sizeStyles[size].fontSize,
          },
        ]}
      >
        {statusLabel}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  text: {
    fontWeight: '600',
    textTransform: 'capitalize',
  },
});

