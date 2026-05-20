import React from 'react';
import { View, ViewProps, StyleSheet } from 'react-native';
import { useTheme } from '@/contexts/theme-context';

interface ContainerProps extends ViewProps {
  children: React.ReactNode;
  padding?: number;
  margin?: number;
  backgroundColor?: string;
}

export function Container({
  children,
  padding = 20,
  margin = 0,
  backgroundColor,
  style,
  ...props
}: ContainerProps) {
  const { colors } = useTheme();

  return (
    <View
      style={[
        styles.container,
        {
          padding,
          margin,
          backgroundColor: backgroundColor || colors.background,
        },
        style,
      ]}
      {...props}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});


