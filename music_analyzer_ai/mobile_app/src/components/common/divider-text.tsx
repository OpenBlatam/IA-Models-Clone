import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY } from '../../constants/config';
import { Divider } from './divider';

interface DividerTextProps {
  text: string;
  variant?: 'horizontal' | 'vertical';
}

/**
 * Divider with text component
 * Visual separator with label
 */
export function DividerText({ text, variant = 'horizontal' }: DividerTextProps) {
  if (variant === 'vertical') {
    return (
      <View style={styles.verticalContainer}>
        <Divider variant="vertical" />
        <Text style={styles.text}>{text}</Text>
        <Divider variant="vertical" />
      </View>
    );
  }

  return (
    <View style={styles.horizontalContainer}>
      <View style={styles.line} />
      <Text style={styles.text}>{text}</Text>
      <View style={styles.line} />
    </View>
  );
}

const styles = StyleSheet.create({
  horizontalContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: SPACING.md,
  },
  verticalContainer: {
    flexDirection: 'column',
    alignItems: 'center',
    marginHorizontal: SPACING.md,
  },
  line: {
    flex: 1,
    height: 1,
    backgroundColor: COLORS.surfaceLight,
  },
  text: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
    marginHorizontal: SPACING.md,
  },
});

