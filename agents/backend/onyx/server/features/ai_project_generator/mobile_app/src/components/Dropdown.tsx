import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, FlatList } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface DropdownOption {
  label: string;
  value: string;
  icon?: React.ReactNode;
  disabled?: boolean;
}

interface DropdownProps {
  options: DropdownOption[];
  selectedValue?: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  label?: string;
  error?: string;
  disabled?: boolean;
  searchable?: boolean;
}

export const Dropdown: React.FC<DropdownProps> = ({
  options,
  selectedValue,
  onValueChange,
  placeholder = 'Seleccionar...',
  label,
  error,
  disabled = false,
  searchable = false,
}) => {
  const { theme } = useTheme();
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const selectedOption = options.find((opt) => opt.value === selectedValue);

  const filteredOptions = searchable
    ? options.filter((opt) =>
        opt.label.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : options;

  const handleSelect = (value: string) => {
    hapticFeedback.selection();
    onValueChange(value);
    setIsOpen(false);
    setSearchQuery('');
  };

  return (
    <View style={styles.container}>
      {label && (
        <Text style={[styles.label, { color: theme.text }]}>{label}</Text>
      )}
      <TouchableOpacity
        style={[
          styles.dropdown,
          {
            backgroundColor: disabled ? theme.surfaceVariant : theme.surface,
            borderColor: error ? theme.error : theme.border,
            opacity: disabled ? 0.6 : 1,
          },
        ]}
        onPress={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        activeOpacity={0.7}
      >
        <Text
          style={[
            styles.dropdownText,
            {
              color: selectedOption
                ? theme.text
                : theme.textTertiary,
            },
          ]}
        >
          {selectedOption ? selectedOption.label : placeholder}
        </Text>
        <Text
          style={[
            styles.arrow,
            {
              color: theme.textSecondary,
              transform: [{ rotate: isOpen ? '180deg' : '0deg' }],
            },
          ]}
        >
          ▼
        </Text>
      </TouchableOpacity>

      {isOpen && (
        <View
          style={[
            styles.optionsContainer,
            {
              backgroundColor: theme.surface,
              borderColor: theme.border,
            },
          ]}
        >
          <FlatList
            data={filteredOptions}
            keyExtractor={(item) => item.value}
            renderItem={({ item }) => (
              <TouchableOpacity
                style={[
                  styles.option,
                  {
                    backgroundColor:
                      item.value === selectedValue
                        ? theme.surfaceVariant
                        : 'transparent',
                    opacity: item.disabled ? 0.5 : 1,
                  },
                ]}
                onPress={() => !item.disabled && handleSelect(item.value)}
                disabled={item.disabled}
                activeOpacity={0.7}
              >
                {item.icon && (
                  <View style={styles.optionIcon}>{item.icon}</View>
                )}
                <Text
                  style={[
                    styles.optionText,
                    {
                      color:
                        item.value === selectedValue
                          ? theme.primary
                          : theme.text,
                      fontWeight:
                        item.value === selectedValue ? '600' : '400',
                    },
                  ]}
                >
                  {item.label}
                </Text>
                {item.value === selectedValue && (
                  <Text style={[styles.checkmark, { color: theme.primary }]}>
                    ✓
                  </Text>
                )}
              </TouchableOpacity>
            )}
            maxHeight={200}
          />
        </View>
      )}

      {error && (
        <Text style={[styles.errorText, { color: theme.error }]}>
          {error}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
    position: 'relative',
    zIndex: 1,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  dropdown: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
  },
  dropdownText: {
    ...typography.body,
    flex: 1,
  },
  arrow: {
    fontSize: 12,
    marginLeft: spacing.sm,
  },
  optionsContainer: {
    position: 'absolute',
    top: '100%',
    left: 0,
    right: 0,
    marginTop: spacing.xs,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    maxHeight: 200,
    zIndex: 1000,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 5,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
  },
  optionIcon: {
    marginRight: spacing.sm,
  },
  optionText: {
    ...typography.body,
    flex: 1,
  },
  checkmark: {
    fontSize: 18,
    fontWeight: '600',
  },
  errorText: {
    ...typography.caption,
    marginTop: spacing.xs,
  },
});

