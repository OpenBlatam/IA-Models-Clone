import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { BlurView as ExpoBlurView } from 'expo-blur';
import { useTheme } from '../context/ThemeContext';

interface BlurViewProps {
  children?: React.ReactNode;
  intensity?: number;
  tint?: 'light' | 'dark' | 'default';
  style?: ViewStyle;
}

const BlurView: React.FC<BlurViewProps> = ({
  children,
  intensity = 50,
  tint = 'default',
  style,
}) => {
  const { isDark } = useTheme();
  const blurTint = tint === 'default' ? (isDark ? 'dark' : 'light') : tint;

  return (
    <ExpoBlurView
      intensity={intensity}
      tint={blurTint}
      style={[styles.container, style]}
    >
      {children}
    </ExpoBlurView>
  );
};

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
  },
});

export default BlurView;

