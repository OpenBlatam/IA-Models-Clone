import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing } from '../theme/colors';

interface DividerProps {
  orientation?: 'horizontal' | 'vertical';
  spacing?: number;
  style?: ViewStyle;
}

export const Divider: React.FC<DividerProps> = ({
  orientation = 'horizontal',
  spacing: dividerSpacing = spacing.md,
  style,
}) => {
  const { theme } = useTheme();

  return (
    <View
      style={[
        orientation === 'horizontal'
          ? [styles.horizontal, { marginVertical: dividerSpacing }]
          : [styles.vertical, { marginHorizontal: dividerSpacing }],
        {
          backgroundColor: theme.border,
        },
        style,
      ]}
    />
  );
};

const styles = StyleSheet.create({
  horizontal: {
    height: 1,
    width: '100%',
  },
  vertical: {
    width: 1,
    height: '100%',
  },
});

