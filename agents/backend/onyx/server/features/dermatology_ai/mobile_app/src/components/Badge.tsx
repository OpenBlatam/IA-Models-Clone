import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface BadgeProps {
  label: string;
  color?: string;
  backgroundColor?: string;
  size?: 'small' | 'medium' | 'large';
}

const Badge: React.FC<BadgeProps> = ({
  label,
  color = '#fff',
  backgroundColor = '#6366f1',
  size = 'medium',
}) => {
  const sizeStyles = {
    small: { paddingHorizontal: 8, paddingVertical: 4, fontSize: 10 },
    medium: { paddingHorizontal: 12, paddingVertical: 6, fontSize: 12 },
    large: { paddingHorizontal: 16, paddingVertical: 8, fontSize: 14 },
  };

  return (
    <View
      style={[
        styles.badge,
        {
          backgroundColor,
          ...sizeStyles[size],
        },
      ]}
    >
      <Text style={[styles.label, { color }]}>{label}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  badge: {
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  label: {
    fontWeight: '600',
  },
});

export default Badge;

