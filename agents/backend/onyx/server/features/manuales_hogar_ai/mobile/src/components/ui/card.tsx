/**
 * Card Component
 * ==============
 * Reusable card component
 */

import { View, StyleSheet, ViewStyle } from 'react-native';
import { ReactNode } from 'react';
import { useApp } from '@/lib/context/app-context';

interface CardProps {
  children: ReactNode;
  style?: ViewStyle;
  padding?: number;
  elevated?: boolean;
}

export function Card({ children, style, padding = 16, elevated = true }: CardProps) {
  const { state } = useApp();
  const colors = state.colors;

  return (
    <View
      style={[
        styles.card,
        {
          backgroundColor: colors.card,
          padding,
          ...(elevated && {
            shadowColor: colors.cardShadow,
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 4,
            elevation: 3,
          }),
        },
        style,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 12,
  },
});



