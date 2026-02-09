import React from 'react';
import { View, StyleSheet, ActivityIndicator } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface SpinnerProps {
  size?: 'small' | 'large';
  color?: string;
  overlay?: boolean;
}

const Spinner: React.FC<SpinnerProps> = ({
  size = 'large',
  color,
  overlay = false,
}) => {
  const { colors } = useTheme();
  const spinnerColor = color || colors.primary;

  if (overlay) {
    return (
      <View style={styles.overlay}>
        <ActivityIndicator size={size} color={spinnerColor} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ActivityIndicator size={size} color={spinnerColor} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
});

export default Spinner;

