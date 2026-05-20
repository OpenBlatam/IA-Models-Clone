import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import LoadingSpinner from './LoadingSpinner';

interface LoadingStateProps {
  message?: string;
  size?: 'small' | 'large';
}

const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Cargando...',
  size = 'large',
}) => {
  const { colors } = useTheme();

  return (
    <View style={styles.container}>
      <LoadingSpinner size={size} color={colors.primary} />
      {message && (
        <Text style={[styles.message, { color: colors.textSecondary }]}>
          {message}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  message: {
    marginTop: 16,
    fontSize: 14,
    textAlign: 'center',
  },
});

export default LoadingState;

