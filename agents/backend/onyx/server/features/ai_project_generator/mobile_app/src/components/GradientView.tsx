import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
// @ts-ignore - expo-linear-gradient types
import { LinearGradient } from 'expo-linear-gradient';

interface GradientViewProps {
  children?: React.ReactNode;
  colors: string[];
  start?: { x: number; y: number };
  end?: { x: number; y: number };
  style?: ViewStyle;
}

export const GradientView: React.FC<GradientViewProps> = ({
  children,
  colors,
  start = { x: 0, y: 0 },
  end = { x: 1, y: 1 },
  style,
}) => {
  return (
    <LinearGradient
      colors={colors}
      start={start}
      end={end}
      style={[styles.container, style]}
    >
      {children}
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

