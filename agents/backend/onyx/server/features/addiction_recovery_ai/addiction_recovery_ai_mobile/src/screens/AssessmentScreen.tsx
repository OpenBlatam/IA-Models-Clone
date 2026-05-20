import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
} from 'react-native';
import { Input, Button, LoadingSpinner } from '@/components';
import { useAuthStore } from '@/store/auth-store';
import { apiService } from '@/services/api';
import { Picker } from '@react-native-picker/picker';
import { AssessmentResponse } from '@/types';

export const AssessmentScreen: React.FC = () => {
  const { user } = useAuthStore();
  const [addictionType, setAddictionType] = useState<'smoking' | 'alcohol' | 'drugs' | 'gambling' | 'internet' | 'other'>('smoking');
  const [severity, setSeverity] = useState<'low' | 'moderate' | 'high' | 'severe'>('moderate');
  const [frequency, setFrequency] = useState<'daily' | 'weekly' | 'monthly' | 'occasional'>('daily');
  const [durationYears, setDurationYears] = useState('');
  const [dailyCost, setDailyCost] = useState('');
  const [previousAttempts, setPreviousAttempts] = useState('0');
  const [supportSystem, setSupportSystem] = useState(false);
  const [loading, setLoading] = useState(false);
  const [assessment, setAssessment] = useState<AssessmentResponse | null>(null);

  const handleAssessment = async () => {
    if (!user?.user_id) {
      Alert.alert('Error', 'Usuario no identificado');
      return;
    }

    setLoading(true);
    try {
      const result = await apiService.assess({
        addiction_type: addictionType,
        severity,
        frequency,
        duration_years: durationYears ? parseFloat(durationYears) : undefined,
        daily_cost: dailyCost ? parseFloat(dailyCost) : undefined,
        previous_attempts: parseInt(previousAttempts, 10) || 0,
        support_system: supportSystem,
        triggers: [],
        motivations: [],
        medical_conditions: [],
      });
      setAssessment(result);
    } catch (error: any) {
      Alert.alert('Error', error.response?.data?.detail || 'Error al realizar evaluación');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Evaluación de Adicción</Text>
        <Text style={styles.subtitle}>
          Completa esta evaluación para obtener un plan personalizado
        </Text>
      </View>

      {!assessment ? (
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
            onPress={handleAssessment}
            loading={loading}
            style={styles.button}
          />
        </View>
      ) : (
        <View style={styles.results}>
          <Text style={styles.resultsTitle}>Resultados de la Evaluación</Text>

          <View style={styles.resultCard}>
            <Text style={styles.resultLabel}>Puntuación de Severidad</Text>
            <Text style={styles.resultValue}>
              {assessment.severity_score.toFixed(1)}/10
            </Text>
          </View>

          <View style={styles.resultCard}>
            <Text style={styles.resultLabel}>Nivel de Riesgo</Text>
            <Text
              style={[
                styles.resultValue,
                {
                  color:
                    assessment.risk_level === 'low'
                      ? '#34C759'
                      : assessment.risk_level === 'moderate'
                      ? '#FF9500'
                      : assessment.risk_level === 'high'
                      ? '#FF3B30'
                      : '#8E8E93',
                },
              ]}
            >
              {assessment.risk_level === 'low'
                ? 'Bajo'
                : assessment.risk_level === 'moderate'
                ? 'Moderado'
                : assessment.risk_level === 'high'
                ? 'Alto'
                : 'Crítico'}
            </Text>
          </View>

          {assessment.recommendations.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Recomendaciones</Text>
              {assessment.recommendations.map((rec, index) => (
                <Text key={index} style={styles.recommendationItem}>
                  • {rec}
                </Text>
              ))}
            </View>
          )}

          {assessment.next_steps.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Próximos Pasos</Text>
              {assessment.next_steps.map((step, index) => (
                <Text key={index} style={styles.recommendationItem}>
                  {index + 1}. {step}
                </Text>
              ))}
            </View>
          )}

          <Button
            title="Nueva Evaluación"
            onPress={() => setAssessment(null)}
            variant="outline"
            style={styles.button}
          />
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    padding: 24,
    backgroundColor: '#FFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
  },
  form: {
    padding: 16,
  },
  pickerContainer: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  pickerWrapper: {
    borderWidth: 1,
    borderColor: '#DDD',
    borderRadius: 8,
    overflow: 'hidden',
  },
  picker: {
    backgroundColor: '#FFF',
  },
  input: {
    marginBottom: 16,
  },
  button: {
    marginTop: 8,
  },
  results: {
    padding: 16,
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  resultCard: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  resultValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  section: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  recommendationItem: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
    lineHeight: 20,
  },
});

