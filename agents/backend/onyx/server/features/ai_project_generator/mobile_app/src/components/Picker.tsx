import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, FlatList, Modal } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface PickerOption {
  label: string;
  value: string;
  icon?: React.ReactNode;
}

interface PickerProps {
  options: PickerOption[];
  selectedValue?: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  label?: string;
  searchable?: boolean;
}

export const Picker: React.FC<PickerProps> = ({
  options,
  selectedValue,
  onValueChange,
  placeholder = 'Seleccionar...',
  label,
  searchable = false,
}) => {
  const { theme } = useTheme();
  const [visible, setVisible] = useState(false);
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
    setVisible(false);
    setSearchQuery('');
  };

  return (
    <View style={styles.container}>
      {label && (
        <Text style={[styles.label, { color: theme.text }]}>{label}</Text>
      )}
      <TouchableOpacity
        style={[
          styles.picker,
          {
            backgroundColor: theme.surface,
            borderColor: theme.border,
          },
        ]}
        onPress={() => setVisible(true)}
        activeOpacity={0.7}
      >
        <Text
          style={[
            styles.pickerText,
            {
              color: selectedOption ? theme.text : theme.textTertiary,
            },
          ]}
        >
          {selectedOption ? selectedOption.label : placeholder}
        </Text>
        <Text style={[styles.arrow, { color: theme.textSecondary }]}>▼</Text>
      </TouchableOpacity>

      <Modal
        visible={visible}
        transparent
        animationType="slide"
        onRequestClose={() => setVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <TouchableOpacity
            style={styles.modalBackdrop}
            activeOpacity={1}
            onPress={() => setVisible(false)}
          />
          <View
            style={[
              styles.modalContent,
              {
                backgroundColor: theme.surface,
              },
            ]}
          >
            <View style={[styles.modalHeader, { borderBottomColor: theme.border }]}>
              <Text style={[styles.modalTitle, { color: theme.text }]}>
                {label || 'Seleccionar'}
              </Text>
              <TouchableOpacity onPress={() => setVisible(false)}>
                <Text style={[styles.closeButton, { color: theme.text }]}>✕</Text>
              </TouchableOpacity>
            </View>
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
                    },
                  ]}
                  onPress={() => handleSelect(item.value)}
                  activeOpacity={0.7}
                >
                  {item.icon && <View style={styles.optionIcon}>{item.icon}</View>}
                  <Text
                    style={[
                      styles.optionText,
                      {
                        color:
                          item.value === selectedValue
                            ? theme.primary
                            : theme.text,
                        fontWeight: item.value === selectedValue ? '600' : '400',
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
            />
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  picker: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
  },
  pickerText: {
    ...typography.body,
    flex: 1,
  },
  arrow: {
    fontSize: 12,
    marginLeft: spacing.sm,
  },
  modalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    maxHeight: '70%',
    borderTopLeftRadius: borderRadius.xl,
    borderTopRightRadius: borderRadius.xl,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.xl,
    borderBottomWidth: 1,
  },
  modalTitle: {
    ...typography.h3,
  },
  closeButton: {
    fontSize: 20,
    fontWeight: '600',
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
    paddingHorizontal: spacing.xl,
  },
  optionIcon: {
    marginRight: spacing.md,
  },
  optionText: {
    ...typography.body,
    flex: 1,
  },
  checkmark: {
    fontSize: 18,
    fontWeight: '600',
  },
});

