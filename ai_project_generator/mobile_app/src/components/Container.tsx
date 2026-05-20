import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing } from '../theme/colors';

interface ContainerProps {
  children: React.ReactNode;
  padding?: boolean;
  paddingHorizontal?: boolean;
  paddingVertical?: boolean;
  margin?: boolean;
  backgroundColor?: string;
  style?: ViewStyle;
}

export const Container: React.FC<ContainerProps> = ({
  children,
  padding = false,
  paddingHorizontal = false,
  paddingVertical = false,
  margin = false,
  backgroundColor,
  style,
}) => {
  const { theme } = useTheme();

  return (
    <View
      style={[
        styles.container,
        padding && { padding: spacing.md },
        paddingHorizontal && { paddingHorizontal: spacing.md },
        paddingVertical && { paddingVertical: spacing.md },
        margin && { margin: spacing.md },
        backgroundColor && { backgroundColor },
        !backgroundColor && { backgroundColor: theme.background },
        style,
      ]}
    >
      {children}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

