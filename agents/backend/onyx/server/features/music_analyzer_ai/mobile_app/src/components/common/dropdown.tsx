import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  FlatList,
} from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface DropdownOption<T> {
  label: string;
  value: T;
  disabled?: boolean;
}

interface DropdownProps<T> {
  options: DropdownOption<T>[];
  value: T;
  onValueChange: (value: T) => void;
  placeholder?: string;
  disabled?: boolean;
  label?: string;
}

/**
 * Dropdown component
 * Custom dropdown with modal
 */
export function Dropdown<T>({
  options,
  value,
  onValueChange,
  placeholder = 'Select...',
  disabled = false,
  label,
}: DropdownProps<T>) {
  const [visible, setVisible] = useState(false);
  const selectedOption = options.find((opt) => opt.value === value);

  const handleSelect = (option: DropdownOption<T>) => {
    if (!option.disabled) {
      onValueChange(option.value);
      setVisible(false);
    }
  };

  return (
    <View style={styles.container}>
      {label && <Text style={styles.label}>{label}</Text>}
      <TouchableOpacity
        style={[styles.button, disabled && styles.disabled]}
        onPress={() => !disabled && setVisible(true)}
        disabled={disabled}
        accessibilityRole="button"
        accessibilityLabel={label || placeholder}
      >
        <Text
          style={[
            styles.buttonText,
            !selectedOption && styles.placeholderText,
          ]}
        >
          {selectedOption ? selectedOption.label : placeholder}
        </Text>
        <Text style={styles.arrow}>▼</Text>
      </TouchableOpacity>

      <Modal
        visible={visible}
        transparent
        animationType="fade"
        onRequestClose={() => setVisible(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setVisible(false)}
        >
          <View style={styles.modalContent}>
            <FlatList
              data={options}
              keyExtractor={(item, index) => `${item.label}-${index}`}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={[
                    styles.option,
                    item.value === value && styles.selectedOption,
                    item.disabled && styles.disabledOption,
                  ]}
                  onPress={() => handleSelect(item)}
                  disabled={item.disabled}
                >
                  <Text
                    style={[
                      styles.optionText,
                      item.value === value && styles.selectedOptionText,
                      item.disabled && styles.disabledOptionText,
                    ]}
                  >
                    {item.label}
                  </Text>
                </TouchableOpacity>
              )}
            />
          </View>
        </TouchableOpacity>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  label: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.text,
    marginBottom: SPACING.xs,
  },
  button: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: COLORS.surfaceLight,
    borderRadius: BORDER_RADIUS.md,
    padding: SPACING.md,
    borderWidth: 1,
    borderColor: COLORS.surfaceLight,
  },
  disabled: {
    opacity: 0.5,
  },
  buttonText: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    flex: 1,
  },
  placeholderText: {
    color: COLORS.textSecondary,
  },
  arrow: {
    color: COLORS.textSecondary,
    marginLeft: SPACING.sm,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.md,
    width: '80%',
    maxHeight: '60%',
  },
  option: {
    padding: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  selectedOption: {
    backgroundColor: COLORS.primary + '20',
  },
  disabledOption: {
    opacity: 0.5,
  },
  optionText: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
  },
  selectedOptionText: {
    color: COLORS.primary,
    fontWeight: '600',
  },
  disabledOptionText: {
    color: COLORS.textSecondary,
  },
});

