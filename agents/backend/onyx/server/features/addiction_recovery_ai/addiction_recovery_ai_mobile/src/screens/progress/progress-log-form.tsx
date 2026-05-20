import React from 'react';
import { View, Text } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { Input, Button } from '@/components';
import { useColors } from '@/theme/colors';
import { useProgressStyles } from './progress-screen.styles';
import { useProgressLogForm } from './use-progress-log-form';

export function ProgressLogForm(): JSX.Element {
  const colors = useColors();
  const styles = useProgressStyles(colors);
  const {
    showLogForm,
    mood,
    setMood,
    cravingsLevel,
    setCravingsLevel,
    notes,
    setNotes,
    handleLogEntry,
    isLoading,
  } = useProgressLogForm();

  if (!showLogForm) {
    return null;
  }

  return (
    <View style={styles.logForm}>
      <Text style={styles.formTitle}>Registrar Entrada del Día</Text>

      <View style={styles.pickerContainer}>
        <Text style={styles.label}>Estado de Ánimo</Text>
        <View style={styles.pickerWrapper}>
          <Picker
            selectedValue={mood}
            onValueChange={(value) => setMood(value)}
            style={styles.picker}
          >
            <Picker.Item label="Excelente" value="excellent" />
            <Picker.Item label="Bueno" value="good" />
            <Picker.Item label="Neutral" value="neutral" />
            <Picker.Item label="Malo" value="poor" />
            <Picker.Item label="Terrible" value="terrible" />
          </Picker>
        </View>
      </View>

      <Input
        label="Nivel de Ansias (0-10)"
        value={cravingsLevel}
        onChangeText={setCravingsLevel}
        keyboardType="number-pad"
        placeholder="5"
        style={styles.input}
      />

      <Input
        label="Notas (Opcional)"
        value={notes}
        onChangeText={setNotes}
        placeholder="¿Cómo te sientes hoy?"
        multiline
        numberOfLines={4}
        style={styles.input}
      />

      <Button
        title="Guardar Entrada"
        onPress={handleLogEntry}
        loading={isLoading}
        style={styles.saveButton}
      />
    </View>
  );
}

