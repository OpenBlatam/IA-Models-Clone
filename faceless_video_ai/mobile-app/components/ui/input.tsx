import React from 'react';
import { TextInput, Text, View, StyleSheet, TextInputProps } from 'react-native';
import { useColorScheme } from 'react-native';

interface InputProps extends TextInputProps {
  label?: string;
  error?: string;
  containerStyle?: object;
}

export function Input({
  label,
  error,
  containerStyle,
  style,
  ...props
}: InputProps) {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  return (
    <View style={[styles.container, containerStyle]}>
      {label && <Text style={[styles.label, isDark && styles.labelDark]}>{label}</Text>}
      <TextInput
        style={[
          styles.input,
          isDark && styles.inputDark,
          error && styles.inputError,
          style,
        ]}
        placeholderTextColor={isDark ? '#8E8E93' : '#8E8E93'}
        {...props}
      />
      {error && <Text style={styles.errorText}>{error}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
    color: '#000000',
  },
  labelDark: {
    color: '#FFFFFF',
  },
  input: {
    borderWidth: 1,
    borderColor: '#C7C7CC',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    backgroundColor: '#FFFFFF',
    color: '#000000',
  },
  inputDark: {
    backgroundColor: '#1C1C1E',
    borderColor: '#38383A',
    color: '#FFFFFF',
  },
  inputError: {
    borderColor: '#FF3B30',
  },
  errorText: {
    fontSize: 12,
    color: '#FF3B30',
    marginTop: 4,
  },
});


