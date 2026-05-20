import React, { ReactNode } from 'react';
import { View, AccessibilityInfo } from 'react-native';

interface AccessibilityWrapperProps {
  children: ReactNode;
  accessible?: boolean;
  accessibilityLabel?: string;
  accessibilityHint?: string;
  accessibilityRole?: string;
  testID?: string;
}

const AccessibilityWrapper: React.FC<AccessibilityWrapperProps> = ({
  children,
  accessible = true,
  accessibilityLabel,
  accessibilityHint,
  accessibilityRole,
  testID,
}) => {
  return (
    <View
      accessible={accessible}
      accessibilityLabel={accessibilityLabel}
      accessibilityHint={accessibilityHint}
      accessibilityRole={accessibilityRole as any}
      testID={testID}
    >
      {children}
    </View>
  );
};

export default AccessibilityWrapper;

