import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import Spinner from './Spinner';

interface LoadingOverlayProps {
  visible: boolean;
  message?: string;
  opacity?: number;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  visible,
  message,
  opacity = 0.7,
}) => {
  const { colors } = useTheme();

  if (!visible) return null;

  return (
    <View
      style={[
        styles.overlay,
        {
          backgroundColor: `rgba(0, 0, 0, ${opacity})`,
        },
      ]}
    >
      <View
        style={[
          styles.content,
          {
            backgroundColor: colors.card,
          },
        ]}
      >
        <Spinner size="large" />
        {message && (
          <Text style={[styles.message, { color: colors.text }]}>
            {message}
          </Text>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  content: {
    padding: 24,
    borderRadius: 12,
    alignItems: 'center',
    minWidth: 120,
  },
  message: {
    marginTop: 16,
    fontSize: 14,
    textAlign: 'center',
  },
});

export default LoadingOverlay;

