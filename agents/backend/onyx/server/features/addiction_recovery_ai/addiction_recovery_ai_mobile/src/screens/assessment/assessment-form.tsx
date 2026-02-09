import React, { useState } from 'react';
import { View, Text } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { Input, Button } from '@/components';
import { useColors } from '@/theme/colors';
import type { AddictionType, SeverityLevel, Frequency } from '@/types';
import { useAssessmentStyles } from './assessment-screen.styles';

interface AssessmentFormProps {
  onSubmit: (data: {
    addictionType: AddictionType;
    severity: SeverityLevel;
    frequency: Frequency;
    durationYears?: string;
    dailyCost?: string;
    previousAttempts: string;
    supportSystem: boolean;
  }) => void;
}

export function AssessmentForm({ onSubmit }: AssessmentFormProps): JSX.Element {
  const colors = useColors();
  const styles = useAssessmentStyles(colors);
  const [addictionType, setAddictionType] = useState<AddictionType>('smoking');
  const [severity, setSeverity] = useState<SeverityLevel>('moderate');
  const [frequency, setFrequency] = useState<Frequency>('daily');
  const [durationYears, setDurationYears] = useState('');
  const [dailyCost, setDailyCost] = useState('');
  const [previousAttempts, setPreviousAttempts] = useState('0');
  const [supportSystem, setSupportSystem] = useState(false);

  const handleSubmit = () => {
    onSubmit({
      addictionType,
      severity,
      frequency,
      durationYears,
      dailyCost,
      previousAttempts,
      supportSystem,
    });
  };

  return (
    <View style={styles.form}>
      <View style={styles.pickerContainer}>
        <Text style={styles.label}>Tipo de Adicción *</Text>
        <View style={styles.pickerWrapper}>
          <Picker
            selectedValue={addictionType}
            onValueChange={(value) => setAddictionType(value)}
            style={styles.picker}
          >
            <Picker.Item label="Fumar" value="smoking" />
            <Picker.Item label="Alcohol" value="alcohol" />
            <Picker.Item label="Drogas" value="drugs" />
            <Picker.Item label="Juego" value="gambling" />
            <Picker.Item label="Internet" value="internet" />
            <Picker.Item label="Otro" value="other" />
          </Picker>
        </View>
      </View>

      <View style={styles.pickerContainer}>
        <Text style={styles.label}>Severidad *</Text>
        <View style={styles.pickerWrapper}>
          <Picker
            selectedValue={severity}
            onValueChange={(value) => setSeverity(value)}
            style={styles.picker}
          >
            <Picker.Item label="Baja" value="low" />
            <Picker.Item label="Moderada" value="moderate" />
            <Picker.Item label="Alta" value="high" />
            <Picker.Item label="Severa" value="severe" />
          </Picker>
        </View>
      </View>

      <View style={styles.pickerContainer}>
        <Text style={styles.label}>Frecuencia *</Text>
        <View style={styles.pickerWrapper}>
          <Picker
            selectedValue={frequency}
            onValueChange={(value) => setFrequency(value)}
            style={styles.picker}
          >
            <Picker.Item label="Diario" value="daily" />
            <Picker.Item label="Semanal" value="weekly" />
            <Picker.Item label="Mensual" value="monthly" />
            <Picker.Item label="Ocasional" value="occasional" />
          </Picker>
        </View>
      </View>

      <Input
        label="Duración (años)"
        value={durationYears}
        onChangeText={setDurationYears}
        keyboardType="decimal-pad"
        placeholder="Ej: 5"
        style={styles.input}
      />

      <Input
        label="Costo Diario"
        value={dailyCost}
        onChangeText={setDailyCost}
        keyboardType="decimal-pad"
        placeholder="Ej: 10.50"
        style={styles.input}
      />

      <Input
        label="Intentos Previos"
        value={previousAttempts}
        onChangeText={setPreviousAttempts}
        keyboardType="number-pad"
        placeholder="0"
        style={styles.input}
      />

      <Button
        title="Realizar Evaluación"
        onPress={handleSubmit}
        style={styles.button}
      />
    </View>
  );
}

