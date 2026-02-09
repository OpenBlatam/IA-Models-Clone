import React from 'react';
import { View, StyleSheet, ActivityIndicator, Text, Modal } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';
import { BlurView } from './BlurView';

interface LoadingOverlayProps {
  visible: boolean;
  message?: string;
  transparent?: boolean;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  visible,
  message,
  transparent = false,
}) => {
  const { theme } = useTheme();

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      statusBarTranslucent
    >
      <View style={styles.container}>
        {!transparent && (
          <BlurView
            intensity={50}
            tint="dark"
            style={StyleSheet.absoluteFill}
          />
        )}
        <View
          style={[
            styles.content,
            {
              backgroundColor: transparent
                ? 'transparent'
                : theme.surface + 'E6',
            },
          ]}
        >
          <ActivityIndicator size="large" color={theme.primary} />
          {message && (
            <Text style={[styles.message, { color: theme.text }]}>
              {message}
            </Text>
          )}
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  content: {
    padding: spacing.xl,
    borderRadius: 12,
    alignItems: 'center',
    minWidth: 120,
  },
  message: {
    ...typography.body,
    marginTop: spacing.md,
    textAlign: 'center',
  },
});

