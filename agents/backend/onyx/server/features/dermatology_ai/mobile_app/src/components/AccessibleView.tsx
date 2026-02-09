import React from 'react';
import { View, ViewProps } from 'react-native';

interface AccessibleViewProps extends ViewProps {
  label: string;
  hint?: string;
  role?: 'button' | 'text' | 'header' | 'link' | 'image' | 'none';
  children: React.ReactNode;
}

const AccessibleView: React.FC<AccessibleViewProps> = ({
  label,
  hint,
  role = 'none',
  children,
  ...props
}) => {
  return (
    <View
      accessible={true}
      accessibilityLabel={label}
      accessibilityHint={hint}
      accessibilityRole={role}
      {...props}
    >
      {children}
    </View>
  );
};

export default AccessibleView;

