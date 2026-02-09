import React from 'react';
import { SafeAreaView as RNSafeAreaView, StyleSheet, ViewStyle } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

interface SafeAreaViewProps {
  children: React.ReactNode;
  style?: ViewStyle;
  edges?: ('top' | 'bottom' | 'left' | 'right')[];
}

/**
 * Enhanced SafeAreaView component with customizable edges
 */
export const SafeAreaView: React.FC<SafeAreaViewProps> = React.memo(
  ({ children, style, edges = ['top', 'bottom', 'left', 'right'] }) => {
    const insets = useSafeAreaInsets();

    const safeAreaStyle = React.useMemo(() => {
      const padding: ViewStyle = {};
      if (edges.includes('top')) padding.paddingTop = insets.top;
      if (edges.includes('bottom')) padding.paddingBottom = insets.bottom;
      if (edges.includes('left')) padding.paddingLeft = insets.left;
      if (edges.includes('right')) padding.paddingRight = insets.right;
      return padding;
    }, [insets, edges]);

    return (
      <RNSafeAreaView style={[styles.container, safeAreaStyle, style]}>
        {children}
      </RNSafeAreaView>
    );
  }
);

SafeAreaView.displayName = 'SafeAreaView';

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

