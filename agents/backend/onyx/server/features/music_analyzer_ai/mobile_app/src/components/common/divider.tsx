import React from 'react';
import { View, StyleSheet } from 'react-native';
import { COLORS, SPACING } from '../../constants/config';

interface DividerProps {
  vertical?: boolean;
  spacing?: number;
  color?: string;
  thickness?: number;
}

/**
 * Divider component for visual separation
 */
export function Divider({
  vertical = false,
  spacing = SPACING.md,
  color = COLORS.surfaceLight,
  thickness = 1,
}: DividerProps) {
  return (
    <View
      style={[
        styles.container,
        vertical
          ? {
              width: thickness,
              height: '100%',
              marginHorizontal: spacing,
            }
          : {
              height: thickness,
              width: '100%',
              marginVertical: spacing,
            },
        { backgroundColor: color },
      ]}
    />
  );
}

const styles = StyleSheet.create({
  container: {},
});

