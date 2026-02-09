import { TextInput, Text, View, StyleSheet, TextInputProps } from 'react-native';

interface TextAreaProps extends TextInputProps {
  label?: string;
  error?: string;
  helperText?: string;
  rows?: number;
}

export function TextArea({ label, error, helperText, rows = 4, style, ...props }: TextAreaProps) {
  const height = rows * 24 + 24; // Approximate height based on rows

  return (
    <View style={styles.container}>
      {label && <Text style={styles.label}>{label}</Text>}
      <TextInput
        style={[
          styles.textArea,
          { height },
          error && styles.textAreaError,
          style,
        ]}
        placeholderTextColor="#9ca3af"
        multiline
        textAlignVertical="top"
        {...props}
      />
      {error && <Text style={styles.errorText}>{error}</Text>}
      {helperText && !error && <Text style={styles.helperText}>{helperText}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    marginBottom: 8,
  },
  textArea: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingTop: 12,
    fontSize: 16,
    backgroundColor: '#fff',
    color: '#1f2937',
  },
  textAreaError: {
    borderColor: '#ef4444',
  },
  errorText: {
    fontSize: 12,
    color: '#ef4444',
    marginTop: 4,
  },
  helperText: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
});


