import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';

interface BlurViewProps {
  children?: React.ReactNode;
  intensity?: number;
  tint?: 'light' | 'dark' | 'default';
  style?: ViewStyle;
}

export const BlurView: React.FC<BlurViewProps> = ({
  children,
  intensity = 50,
  tint = 'default',
  style,
}) => {
  const { theme } = useTheme();
  
  // Simulación de blur con opacidad
  const opacity = intensity / 100;
  const backgroundColor = tint === 'dark' 
    ? `rgba(0, 0, 0, ${opacity * 0.5})`
    : tint === 'light'
    ? `rgba(255, 255, 255, ${opacity * 0.5})`
    : theme.surfaceVariant;

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor,
        },
        style,
      ]}
    >
      {children}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
  },
});

