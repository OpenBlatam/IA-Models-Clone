import { Text, TextProps, StyleSheet } from 'react-native';

interface AccessibleTextProps extends TextProps {
  children: React.ReactNode;
  variant?: 'heading' | 'subheading' | 'body' | 'caption' | 'label';
  accessibilityRole?: 'text' | 'header' | 'link' | 'none';
}

export function AccessibleText({
  children,
  variant = 'body',
  accessibilityRole,
  style,
  ...props
}: AccessibleTextProps) {
  const getVariantStyle = () => {
    switch (variant) {
      case 'heading':
        return styles.heading;
      case 'subheading':
        return styles.subheading;
      case 'body':
        return styles.body;
      case 'caption':
        return styles.caption;
      case 'label':
        return styles.label;
      default:
        return styles.body;
    }
  };

  const getAccessibilityRole = () => {
    if (accessibilityRole) return accessibilityRole;
    if (variant === 'heading' || variant === 'subheading') return 'header';
    return 'text';
  };

  return (
    <Text
      style={[getVariantStyle(), style]}
      accessibilityRole={getAccessibilityRole()}
      allowFontScaling
      {...props}
    >
      {children}
    </Text>
  );
}

const styles = StyleSheet.create({
  heading: {
    fontSize: 28,
    fontWeight: 'bold',
    lineHeight: 36,
  },
  subheading: {
    fontSize: 20,
    fontWeight: '600',
    lineHeight: 28,
  },
  body: {
    fontSize: 16,
    lineHeight: 24,
  },
  caption: {
    fontSize: 12,
    lineHeight: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
    lineHeight: 20,
  },
});


