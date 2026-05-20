import React, { useState } from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useClipboard } from '../hooks/useClipboard';
import { useTheme } from '../context/ThemeContext';
import { useHapticFeedback } from '../hooks/useHapticFeedback';

interface CopyButtonProps {
  text: string;
  label?: string;
  showLabel?: boolean;
  variant?: 'icon' | 'button' | 'text';
}

const CopyButton: React.FC<CopyButtonProps> = ({
  text,
  label = 'Copiar',
  showLabel = false,
  variant = 'icon',
}) => {
  const { colors } = useTheme();
  const { copyToClipboard } = useClipboard();
  const { trigger } = useHapticFeedback();
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const success = await copyToClipboard(text);
    if (success) {
      setCopied(true);
      trigger('success');
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (variant === 'icon') {
    return (
      <TouchableOpacity
        onPress={handleCopy}
        style={styles.iconButton}
        activeOpacity={0.7}
      >
        <Ionicons
          name={copied ? 'checkmark' : 'copy'}
          size={20}
          color={copied ? colors.success : colors.primary}
        />
      </TouchableOpacity>
    );
  }

  if (variant === 'text') {
    return (
      <TouchableOpacity
        onPress={handleCopy}
        style={styles.textButton}
        activeOpacity={0.7}
      >
        <Text style={[styles.text, { color: colors.primary }]}>
          {copied ? 'Copiado' : label}
        </Text>
      </TouchableOpacity>
    );
  }

  return (
    <TouchableOpacity
      onPress={handleCopy}
      style={[
        styles.button,
        {
          backgroundColor: copied ? colors.success : colors.primary,
        },
      ]}
      activeOpacity={0.8}
    >
      <Ionicons
        name={copied ? 'checkmark' : 'copy'}
        size={16}
        color="#fff"
        style={styles.buttonIcon}
      />
      {showLabel && (
        <Text style={styles.buttonText}>
          {copied ? 'Copiado' : label}
        </Text>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  iconButton: {
    padding: 8,
  },
  textButton: {
    padding: 4,
  },
  text: {
    fontSize: 14,
    fontWeight: '500',
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  buttonIcon: {
    marginRight: 4,
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
});

export default CopyButton;

