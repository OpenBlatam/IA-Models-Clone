import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface FilterOption {
  label: string;
  value: string;
}

interface FilterModalProps {
  visible: boolean;
  onClose: () => void;
  title: string;
  options: FilterOption[];
  selectedValues: string[];
  onApply: (values: string[]) => void;
  multiSelect?: boolean;
}

const FilterModal: React.FC<FilterModalProps> = ({
  visible,
  onClose,
  title,
  options,
  selectedValues,
  onApply,
  multiSelect = true,
}) => {
  const [tempSelected, setTempSelected] = useState<string[]>(selectedValues);

  const toggleSelection = (value: string) => {
    if (multiSelect) {
      setTempSelected(prev =>
        prev.includes(value)
          ? prev.filter(v => v !== value)
          : [...prev, value]
      );
    } else {
      setTempSelected([value]);
    }
  };

  const handleApply = () => {
    onApply(tempSelected);
    onClose();
  };

  const handleReset = () => {
    setTempSelected([]);
    onApply([]);
    onClose();
  };

  return (
    <Modal
      visible={visible}
      transparent={true}
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <View style={styles.modal}>
          <View style={styles.header}>
            <Text style={styles.title}>{title}</Text>
            <TouchableOpacity onPress={onClose}>
              <Ionicons name="close" size={24} color="#1f2937" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.content}>
            {options.map((option) => (
              <TouchableOpacity
                key={option.value}
                style={[
                  styles.option,
                  tempSelected.includes(option.value) && styles.optionSelected,
                ]}
                onPress={() => toggleSelection(option.value)}
              >
                <Text
                  style={[
                    styles.optionText,
                    tempSelected.includes(option.value) && styles.optionTextSelected,
                  ]}
                >
                  {option.label}
                </Text>
                {tempSelected.includes(option.value) && (
                  <Ionicons name="checkmark" size={20} color="#6366f1" />
                )}
              </TouchableOpacity>
            ))}
          </ScrollView>

          <View style={styles.footer}>
            <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
              <Text style={styles.resetButtonText}>Limpiar</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.applyButton} onPress={handleApply}>
              <Text style={styles.applyButtonText}>Aplicar</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modal: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  content: {
    padding: 20,
    maxHeight: 400,
  },
  option: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    marginBottom: 8,
    backgroundColor: '#f9fafb',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  optionSelected: {
    backgroundColor: '#f3f4f6',
    borderColor: '#6366f1',
  },
  optionText: {
    fontSize: 16,
    color: '#1f2937',
  },
  optionTextSelected: {
    fontWeight: '600',
    color: '#6366f1',
  },
  footer: {
    flexDirection: 'row',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    gap: 12,
  },
  resetButton: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#e5e7eb',
    alignItems: 'center',
  },
  resetButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6b7280',
  },
  applyButton: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#6366f1',
    alignItems: 'center',
  },
  applyButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
});

export default FilterModal;

