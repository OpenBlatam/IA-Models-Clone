import React from 'react';
import { TouchableOpacity, StyleSheet, Text } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useCopyToClipboard } from '../../hooks/use-copy-to-clipboard';

interface CopyButtonProps {
  text: string;
  label?: string;
  onCopy?: () => void;
}

/**
 * Copy button component
 * Copies text to clipboard
 */
export function CopyButton({ text, label = 'Copy', onCopy }: CopyButtonProps) {
  const { copy, copied } = useCopyToClipboard();

  const handlePress = async () => {
    const success = await copy(text);
    if (success && onCopy) {
      onCopy();
    }
  };

  return (
    <TouchableOpacity
      style={[styles.button, copied && styles.buttonCopied]}
      onPress={handlePress}
      accessibilityLabel={copied ? 'Copied' : label}
      accessibilityRole="button"
    >
      <Text style={[styles.text, copied && styles.textCopied]}>
        {copied ? 'Copied!' : label}
      </Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    paddingVertical: SPACING.sm,
    paddingHorizontal: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonCopied: {
    backgroundColor: COLORS.success,
  },
  text: {
    ...TYPOGRAPHY.body,
    color: COLORS.surface,
    fontWeight: '600',
  },
  textCopied: {
    color: COLORS.surface,
  },
});

