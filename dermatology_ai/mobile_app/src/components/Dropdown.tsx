import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  FlatList,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface DropdownOption {
  label: string;
  value: string;
  icon?: keyof typeof Ionicons.glyphMap;
}

interface DropdownProps {
  options: DropdownOption[];
  value?: string;
  onSelect: (value: string) => void;
  placeholder?: string;
  label?: string;
  error?: string;
}

const Dropdown: React.FC<DropdownProps> = ({
  options,
  value,
  onSelect,
  placeholder = 'Seleccionar...',
  label,
  error,
}) => {
  const { colors } = useTheme();
  const [isOpen, setIsOpen] = useState(false);
  const selectedOption = options.find((opt) => opt.value === value);

  return (
    <View style={styles.container}>
      {label && (
        <Text style={[styles.label, { color: colors.text }]}>{label}</Text>
      )}
      <TouchableOpacity
        style={[
          styles.dropdown,
          {
            backgroundColor: colors.surface,
            borderColor: error ? colors.error : colors.border,
          },
        ]}
        onPress={() => setIsOpen(true)}
        activeOpacity={0.7}
      >
        <Text
          style={[
            styles.selectedText,
            {
              color: selectedOption ? colors.text : colors.textSecondary,
            },
          ]}
        >
          {selectedOption?.label || placeholder}
        </Text>
        <Ionicons
          name="chevron-down"
          size={20}
          color={colors.textSecondary}
        />
      </TouchableOpacity>
      {error && (
        <Text style={[styles.error, { color: colors.error }]}>{error}</Text>
      )}

      <Modal
        visible={isOpen}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setIsOpen(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setIsOpen(false)}
        >
          <View
            style={[styles.modalContent, { backgroundColor: colors.card }]}
            onStartShouldSetResponder={() => true}
          >
            <View style={styles.modalHeader}>
              <Text style={[styles.modalTitle, { color: colors.text }]}>
                {label || 'Seleccionar'}
              </Text>
              <TouchableOpacity onPress={() => setIsOpen(false)}>
                <Ionicons name="close" size={24} color={colors.text} />
              </TouchableOpacity>
            </View>
            <FlatList
              data={options}
              keyExtractor={(item) => item.value}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={[
                    styles.option,
                    {
                      backgroundColor:
                        item.value === value
                          ? `${colors.primary}20`
                          : 'transparent',
                    },
                  ]}
                  onPress={() => {
                    onSelect(item.value);
                    setIsOpen(false);
                  }}
                  activeOpacity={0.7}
                >
                  {item.icon && (
                    <Ionicons
                      name={item.icon}
                      size={20}
                      color={item.value === value ? colors.primary : colors.textSecondary}
                      style={styles.optionIcon}
                    />
                  )}
                  <Text
                    style={[
                      styles.optionText,
                      {
                        color:
                          item.value === value ? colors.primary : colors.text,
                        fontWeight: item.value === value ? '600' : '400',
                      },
                    ]}
                  >
                    {item.label}
                  </Text>
                  {item.value === value && (
                    <Ionicons
                      name="checkmark"
                      size={20}
                      color={colors.primary}
                      style={styles.checkIcon}
                    />
                  )}
                </TouchableOpacity>
              )}
            />
          </View>
        </TouchableOpacity>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 8,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 8,
  },
  dropdown: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 1,
  },
  selectedText: {
    fontSize: 16,
    flex: 1,
  },
  error: {
    fontSize: 12,
    marginTop: 4,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    width: '100%',
    maxWidth: 400,
    borderRadius: 16,
    maxHeight: '80%',
    overflow: 'hidden',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  optionIcon: {
    marginRight: 12,
  },
  optionText: {
    flex: 1,
    fontSize: 16,
  },
  checkIcon: {
    marginLeft: 8,
  },
});

export default Dropdown;

