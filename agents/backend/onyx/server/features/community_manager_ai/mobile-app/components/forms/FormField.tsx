import { ReactNode } from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { Control, Controller, FieldPath, FieldValues } from 'react-hook-form';

interface FormFieldProps<T extends FieldValues> {
  control: Control<T>;
  name: FieldPath<T>;
  label?: string;
  error?: string;
  helperText?: string;
  required?: boolean;
  children: (field: {
    onChange: (value: any) => void;
    onBlur: () => void;
    value: any;
    error?: string;
  }) => ReactNode;
  containerStyle?: ViewStyle;
}

export function FormField<T extends FieldValues>({
  control,
  name,
  label,
  error,
  helperText,
  required,
  children,
  containerStyle,
}: FormFieldProps<T>) {
  return (
    <Controller
      control={control}
      name={name}
      render={({ field: { onChange, onBlur, value }, fieldState: { error: fieldError } }) => {
        const displayError = error || fieldError?.message;

        return (
          <View style={[styles.container, containerStyle]}>
            {label && (
              <Text style={styles.label}>
                {label}
                {required && <Text style={styles.required}> *</Text>}
              </Text>
            )}
            {children({
              onChange,
              onBlur,
              value,
              error: displayError,
            })}
            {displayError && <Text style={styles.errorText}>{displayError}</Text>}
            {helperText && !displayError && <Text style={styles.helperText}>{helperText}</Text>}
          </View>
        );
      }}
    />
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
  required: {
    color: '#ef4444',
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


