import React, { memo, ReactNode } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { useTheme } from '@/theme/theme';

export interface CardProps {
  children: ReactNode;
  style?: ViewStyle;
  padding?: 'none' | 'small' | 'medium' | 'large';
  shadow?: 'none' | 'sm' | 'md' | 'lg';
  accessibilityLabel?: string;
  accessibilityRole?: 'none' | 'button' | 'link' | 'search' | 'image' | 'keyboardkey' | 'text' | 'adjustable' | 'imagebutton' | 'header' | 'summary' | 'alert' | 'checkbox' | 'combobox' | 'menu' | 'menubar' | 'menuitem' | 'progressbar' | 'radio' | 'radiogroup' | 'scrollbar' | 'searchbox' | 'spinbutton' | 'switch' | 'tab' | 'tablist' | 'timer' | 'toolbar';
}

function CardComponent({
  children,
  style,
  padding = 'medium',
  shadow = 'md',
  accessibilityLabel,
  accessibilityRole = 'none',
}: CardProps) {
  const theme = useTheme();

  const cardStyle = [
    styles.card,
    {
      backgroundColor: theme.colors.card,
      borderRadius: theme.borderRadius.lg,
      padding: theme.spacing[padding === 'none' ? 'xs' : padding],
      ...(shadow !== 'none' ? theme.shadows[shadow] : {}),
    },
    style,
  ];

  return (
    <View
      style={cardStyle}
      accessibilityLabel={accessibilityLabel}
      accessibilityRole={accessibilityRole}
    >
      {children}
    </View>
  );
}

export const Card = memo(CardComponent);

const styles = StyleSheet.create({
  card: {
    overflow: 'hidden',
  },
});


