import React from 'react';
import { ScrollView, ScrollViewProps, StyleSheet } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { COLORS } from '../../constants/config';

interface SafeAreaScrollViewProps extends ScrollViewProps {
  children: React.ReactNode;
  edges?: ('top' | 'bottom' | 'left' | 'right')[];
}

/**
 * ScrollView that respects safe area boundaries
 * Follows Expo best practices for safe area management
 */
export function SafeAreaScrollView({
  children,
  edges = ['top', 'bottom'],
  style,
  contentContainerStyle,
  ...props
}: SafeAreaScrollViewProps) {
  const insets = useSafeAreaInsets();

  const paddingStyle = {
    paddingTop: edges.includes('top') ? insets.top : 0,
    paddingBottom: edges.includes('bottom') ? insets.bottom : 0,
    paddingLeft: edges.includes('left') ? insets.left : 0,
    paddingRight: edges.includes('right') ? insets.right : 0,
  };

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: COLORS.background }]}
      edges={[]}
    >
      <ScrollView
        style={[styles.scrollView, style]}
        contentContainerStyle={[
          styles.contentContainer,
          paddingStyle,
          contentContainerStyle,
        ]}
        {...props}
      >
        {children}
      </ScrollView>
    </SafeAreaView>
  );
}

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

