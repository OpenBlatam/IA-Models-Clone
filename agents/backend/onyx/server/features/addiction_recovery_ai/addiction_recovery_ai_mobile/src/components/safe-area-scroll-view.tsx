import React, { memo, ReactNode } from 'react';
import { SafeAreaView, ScrollView, ScrollViewProps, StyleSheet, ViewStyle } from 'react-native';
import { useColors } from '@/theme/colors';

// Types
interface SafeAreaScrollViewProps extends ScrollViewProps {
  children: ReactNode;
  edges?: ('top' | 'bottom' | 'left' | 'right')[];
  style?: ViewStyle;
  contentContainerStyle?: ViewStyle;
}

// Component
function SafeAreaScrollViewComponent({
  children,
  edges = ['top', 'bottom'],
  style,
  contentContainerStyle,
  ...scrollViewProps
}: SafeAreaScrollViewProps): JSX.Element {
  const colors = useColors();

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: colors.background }, style]}
      edges={edges}
    >
      <ScrollView
        {...scrollViewProps}
        style={styles.scrollView}
        contentContainerStyle={[
          styles.contentContainer,
          { backgroundColor: colors.background },
          contentContainerStyle,
        ]}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {children}
      </ScrollView>
    </SafeAreaView>
  );
}

export const SafeAreaScrollView = memo(SafeAreaScrollViewComponent);

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  contentContainer: {
    flexGrow: 1,
  },
});
