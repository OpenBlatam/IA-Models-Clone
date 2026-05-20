import React from 'react';
import { View, ViewProps, AccessibilityProps } from 'react-native';

interface AccessibilityWrapperProps extends ViewProps, AccessibilityProps {
  children: React.ReactNode;
  label: string;
  hint?: string;
  role?: 'button' | 'text' | 'header' | 'link' | 'image' | 'none';
  state?: {
    disabled?: boolean;
    selected?: boolean;
    checked?: boolean;
  };
}

export const AccessibilityWrapper: React.FC<AccessibilityWrapperProps> = ({
  children,
  label,
  hint,
  role = 'none',
  state,
  ...props
}) => {
  return (
    <View
      {...props}
      accessible={true}
      accessibilityLabel={label}
      accessibilityHint={hint}
      accessibilityRole={role}
      accessibilityState={state}
    >
      {children}
    </View>
  );
};

export const withAccessibility = <P extends object>(
  Component: React.ComponentType<P>,
  defaultLabel: string,
  defaultHint?: string
) => {
  return (props: P & { accessibilityLabel?: string; accessibilityHint?: string }) => {
    return (
      <AccessibilityWrapper
        label={props.accessibilityLabel || defaultLabel}
        hint={props.accessibilityHint || defaultHint}
      >
        <Component {...props} />
      </AccessibilityWrapper>
    );
  };
};

