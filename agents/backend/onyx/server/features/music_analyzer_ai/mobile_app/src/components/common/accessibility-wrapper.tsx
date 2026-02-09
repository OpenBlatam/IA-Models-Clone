import React from 'react';
import { View, ViewProps, StyleSheet } from 'react-native';
import { MIN_TOUCH_TARGET, meetsMinimumTouchTarget } from '../../utils/text-scaling';

interface AccessibilityWrapperProps extends ViewProps {
  children: React.ReactNode;
  minTouchTarget?: number;
  accessibilityLabel: string;
  accessibilityRole?: string;
  accessibilityHint?: string;
  accessibilityState?: {
    disabled?: boolean;
    selected?: boolean;
    checked?: boolean;
  };
}

/**
 * Wrapper component that ensures accessibility standards
 * Automatically adjusts touch target size if needed
 */
export function AccessibilityWrapper({
  children,
  minTouchTarget = MIN_TOUCH_TARGET,
  style,
  accessibilityLabel,
  accessibilityRole = 'button',
  accessibilityHint,
  accessibilityState,
  ...props
}: AccessibilityWrapperProps) {
  const [layout, setLayout] = React.useState({ width: 0, height: 0 });

  const onLayout = (event: { nativeEvent: { layout: { width: number; height: number } } }) => {
    const { width, height } = event.nativeEvent.layout;
    setLayout({ width, height });
  };

  const needsPadding =
    !meetsMinimumTouchTarget(layout.width) || !meetsMinimumTouchTarget(layout.height);

  const paddingStyle = needsPadding
    ? {
        paddingHorizontal: Math.max(0, (minTouchTarget - layout.width) / 2),
        paddingVertical: Math.max(0, (minTouchTarget - layout.height) / 2),
      }
    : {};

  return (
    <View
      style={[styles.container, paddingStyle, style]}
      onLayout={onLayout}
      accessible={true}
      accessibilityLabel={accessibilityLabel}
      accessibilityRole={accessibilityRole}
      accessibilityHint={accessibilityHint}
      accessibilityState={accessibilityState}
      {...props}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    minHeight: MIN_TOUCH_TARGET,
    minWidth: MIN_TOUCH_TARGET,
  },
});

