import React, { memo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useColors } from '@/theme/colors';

interface ProgressCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  color?: string;
  accessibilityLabel?: string;
}

function ProgressCardComponent({
  title,
  value,
  subtitle,
  color,
  accessibilityLabel,
}: ProgressCardProps): JSX.Element {
  const colors = useColors();
  const cardColor = color || colors.primary;

  return (
    <View
      style={[
        styles.card,
        {
          backgroundColor: colors.card,
          borderLeftColor: cardColor,
          shadowColor: colors.shadow,
        },
      ]}
      accessibilityRole="text"
      accessibilityLabel={accessibilityLabel || `${title}: ${value}${subtitle ? `, ${subtitle}` : ''}`}
    >
      <Text style={[styles.title, { color: colors.textSecondary }]}>
        {title}
      </Text>
      <Text style={[styles.value, { color: cardColor }]}>{value}</Text>
      {subtitle && (
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
          {subtitle}
        </Text>
      )}
    </View>
  );
}

export const ProgressCard = memo(ProgressCardComponent);

const styles = StyleSheet.create({
  card: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  title: {
    fontSize: 14,
    marginBottom: 4,
  },
  value: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 12,
  },
});

