import { AccessibilityProps } from 'react-native';

export interface AccessibilityConfig {
  label?: string;
  hint?: string;
  role?: AccessibilityProps['accessibilityRole'];
  state?: AccessibilityProps['accessibilityState'];
  value?: string;
  liveRegion?: 'none' | 'polite' | 'assertive';
}

export function createAccessibilityProps(
  config: AccessibilityConfig
): AccessibilityProps {
  const props: AccessibilityProps = {};

  if (config.label) {
    props.accessibilityLabel = config.label;
  }

  if (config.hint) {
    props.accessibilityHint = config.hint;
  }

  if (config.role) {
    props.accessibilityRole = config.role;
  }

  if (config.state) {
    props.accessibilityState = config.state;
  }

  if (config.value !== undefined) {
    props.accessibilityValue = { text: String(config.value) };
  }

  if (config.liveRegion) {
    props.accessibilityLiveRegion = config.liveRegion;
  }

  return props;
}

export function getAccessibilityRole(
  elementType: 'button' | 'link' | 'text' | 'image' | 'header' | 'none'
): AccessibilityProps['accessibilityRole'] {
  const roleMap: Record<string, AccessibilityProps['accessibilityRole']> = {
    button: 'button',
    link: 'link',
    text: 'text',
    image: 'image',
    header: 'header',
    none: 'none',
  };

  return roleMap[elementType] || 'none';
}

