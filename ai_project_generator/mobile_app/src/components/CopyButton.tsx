import React, { useState } from 'react';
import { TouchableOpacity, Text, StyleSheet, ViewStyle, TextStyle } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { copyToClipboard } from '../utils/clipboard';
import { useToastHelpers } from '../hooks/useToast';
import { hapticFeedback } from '../utils/haptics';
import { spacing, borderRadius, typography } from '../theme/colors';

interface CopyButtonProps {
  text: string;
  label?: string;
  showLabel?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
  onCopy?: () => void;
}

export const CopyButton: React.FC<CopyButtonProps> = ({
  text,
  label = 'Copiar',
  showLabel = false,
  style,
  textStyle,
  onCopy,
}) => {
  const { theme } = useTheme();
  const toast = useToastHelpers();
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const success = await copyToClipboard(text);
    if (success) {
      hapticFeedback.success();
      toast.showSuccess('Copiado al portapapeles');
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      onCopy?.();
    } else {
      hapticFeedback.error();
      toast.showError('Error al copiar');
    }
  };

  return (
    <TouchableOpacity
      style={[
        styles.button,
        {
          backgroundColor: copied ? theme.success : theme.surfaceVariant,
          borderColor: theme.border,
        },
        style,
      ]}
      onPress={handleCopy}
      activeOpacity={0.7}
    >
      <Text style={[styles.icon, { color: copied ? theme.surface : theme.text }]}>
        {copied ? '✓' : '📋'}
      </Text>
      {showLabel && (
        <Text
          style={[
            styles.label,
            {
              color: copied ? theme.surface : theme.text,
            },
            textStyle,
          ]}
        >
          {copied ? 'Copiado' : label}
        </Text>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.sm,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    gap: spacing.xs,
  },
  icon: {
    fontSize: 16,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '500',
  },
});

