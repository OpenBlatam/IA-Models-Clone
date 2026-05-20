import React from 'react';
import {
  KeyboardAvoidingView as RNKeyboardAvoidingView,
  Platform,
  StyleSheet,
  ViewStyle,
} from 'react-native';
import { useKeyboard } from '@react-native-community/hooks';

interface KeyboardAvoidingViewProps {
  children: React.ReactNode;
  style?: ViewStyle;
  behavior?: 'padding' | 'height' | 'position';
  enabled?: boolean;
}

/**
 * Enhanced KeyboardAvoidingView with automatic keyboard detection
 */
export const KeyboardAvoidingView: React.FC<KeyboardAvoidingViewProps> =
  React.memo(({ children, style, behavior, enabled = true }) => {
    const keyboard = useKeyboard();

    const defaultBehavior = React.useMemo(
      () => behavior || (Platform.OS === 'ios' ? 'padding' : 'height'),
      [behavior]
    );

    if (!enabled) {
      return <>{children}</>;
    }

    return (
      <RNKeyboardAvoidingView
        style={[styles.container, style]}
        behavior={defaultBehavior}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        {children}
      </RNKeyboardAvoidingView>
    );
  });

KeyboardAvoidingView.displayName = 'KeyboardAvoidingView';

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

