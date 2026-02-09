import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { format } from 'date-fns';

interface DatePickerProps {
  value: Date;
  onChange: (date: Date) => void;
  label?: string;
  minimumDate?: Date;
  maximumDate?: Date;
}

const DatePicker: React.FC<DatePickerProps> = ({
  value,
  onChange,
  label,
  minimumDate,
  maximumDate,
}) => {
  const [modalVisible, setModalVisible] = useState(false);

  const handleDateSelect = (date: Date) => {
    onChange(date);
    setModalVisible(false);
  };

  const generateDateOptions = () => {
    const options: Date[] = [];
    const today = new Date();
    const startDate = minimumDate || new Date(today.getFullYear() - 1, 0, 1);
    const endDate = maximumDate || today;

    for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
      options.push(new Date(d));
    }

    return options.reverse().slice(0, 365); // Last year
  };

  return (
    <>
      <TouchableOpacity
        style={styles.container}
        onPress={() => setModalVisible(true)}
        activeOpacity={0.7}
      >
        {label && <Text style={styles.label}>{label}</Text>}
        <View style={styles.input}>
          <Ionicons name="calendar" size={20} color="#6366f1" />
          <Text style={styles.dateText}>
            {format(value, 'dd MMM yyyy')}
          </Text>
          <Ionicons name="chevron-down" size={20} color="#9ca3af" />
        </View>
      </TouchableOpacity>

      <Modal
        visible={modalVisible}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Seleccionar Fecha</Text>
              <TouchableOpacity onPress={() => setModalVisible(false)}>
                <Ionicons name="close" size={24} color="#1f2937" />
              </TouchableOpacity>
            </View>
            <ScrollView style={styles.dateList}>
              {generateDateOptions().map((date, index) => (
                <TouchableOpacity
                  key={index}
                  style={[
                    styles.dateOption,
                    format(date, 'yyyy-MM-dd') === format(value, 'yyyy-MM-dd') &&
                      styles.dateOptionSelected,
                  ]}
                  onPress={() => handleDateSelect(date)}
                >
                  <Text
                    style={[
                      styles.dateOptionText,
                      format(date, 'yyyy-MM-dd') === format(value, 'yyyy-MM-dd') &&
                        styles.dateOptionTextSelected,
                    ]}
                  >
                    {format(date, 'EEEE, dd MMMM yyyy')}
                  </Text>
                  {format(date, 'yyyy-MM-dd') === format(value, 'yyyy-MM-dd') && (
                    <Ionicons name="checkmark" size={20} color="#6366f1" />
                  )}
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  input: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  dateText: {
    flex: 1,
    fontSize: 16,
    color: '#1f2937',
    marginLeft: 12,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  dateList: {
    padding: 20,
  },
  dateOption: {
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
  dateOptionSelected: {
    backgroundColor: '#f3f4f6',
    borderColor: '#6366f1',
  },
  dateOptionText: {
    fontSize: 16,
    color: '#1f2937',
  },
  dateOptionTextSelected: {
    fontWeight: '600',
    color: '#6366f1',
  },
});

export default DatePicker;

