import React, { ReactNode } from 'react';
import { KeyboardAwareScrollView } from 'react-native-keyboard-aware-scroll-view';
import { StyleSheet, ViewStyle } from 'react-native';
import { useColors } from '@/theme/colors';

interface KeyboardAwareScrollViewProps {
  children: ReactNode;
  style?: ViewStyle;
  contentContainerStyle?: ViewStyle;
  enableOnAndroid?: boolean;
  enableAutomaticScroll?: boolean;
  extraScrollHeight?: number;
}

export function CustomKeyboardAwareScrollView({
  children,
  style,
  contentContainerStyle,
  enableOnAndroid = true,
  enableAutomaticScroll = true,
  extraScrollHeight = 20,
}: KeyboardAwareScrollViewProps): JSX.Element {
  const colors = useColors();

  return (
    <KeyboardAwareScrollView
      style={[styles.container, { backgroundColor: colors.background }, style]}
      contentContainerStyle={[
        styles.contentContainer,
        contentContainerStyle,
      ]}
      enableOnAndroid={enableOnAndroid}
      enableAutomaticScroll={enableAutomaticScroll}
      extraScrollHeight={extraScrollHeight}
      keyboardShouldPersistTaps="handled"
      showsVerticalScrollIndicator={false}
    >
      {children}
    </KeyboardAwareScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  contentContainer: {
    flexGrow: 1,
  },
});

