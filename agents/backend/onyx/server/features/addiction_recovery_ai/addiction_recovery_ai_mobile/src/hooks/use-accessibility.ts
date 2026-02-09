import { useMemo } from 'react';
import { AccessibilityProps } from 'react-native';

interface UseAccessibilityOptions {
  label?: string;
  hint?: string;
  role?: AccessibilityProps['accessibilityRole'];
  state?: AccessibilityProps['accessibilityState'];
  value?: string | number;
  disabled?: boolean;
  selected?: boolean;
  checked?: boolean | 'mixed';
}

export function useAccessibility({
  label,
  hint,
  role = 'none',
  state,
  value,
  disabled = false,
  selected,
  checked,
}: UseAccessibilityOptions): AccessibilityProps {
  return useMemo(
    () => ({
      accessibilityLabel: label,
      accessibilityHint: hint,
      accessibilityRole: role,
      accessibilityValue: value !== undefined ? { text: String(value) } : undefined,
      accessibilityState: {
        disabled,
        selected,
        checked,
        ...state,
      },
    }),
    [label, hint, role, state, value, disabled, selected, checked]
  );
}

