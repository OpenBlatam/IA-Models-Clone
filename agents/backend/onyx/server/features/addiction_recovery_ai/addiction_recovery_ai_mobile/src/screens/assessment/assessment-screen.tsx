import React, { useState } from 'react';
import { View, Text, ScrollView, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Input, Button, LoadingSpinner } from '@/components';
import { useAuthStore } from '@/store/auth-store';
import { assessmentApi } from '@/services/api';
import { useColors } from '@/theme/colors';
import { Picker } from '@react-native-picker/picker';
import type { AssessmentResponse, AddictionType, SeverityLevel, Frequency } from '@/types';
import { useAssessmentStyles } from './assessment-screen.styles';
import { AssessmentForm } from './assessment-form';
import { AssessmentResults } from './assessment-results';

export function AssessmentScreen(): JSX.Element {
  const colors = useColors();
  const { user } = useAuthStore();
  const [loading, setLoading] = useState(false);
  const [assessment, setAssessment] = useState<AssessmentResponse | null>(null);
  const styles = useAssessmentStyles(colors);

  const handleAssessment = async (
    formData: {
      addictionType: AddictionType;
      severity: SeverityLevel;
      frequency: Frequency;
      durationYears?: string;
      dailyCost?: string;
      previousAttempts: string;
      supportSystem: boolean;
    }
  ) => {
    if (!user?.user_id) {
      Alert.alert('Error', 'Usuario no identificado');
      return;
    }

    setLoading(true);
    try {
      const result = await assessmentApi.assess({
        addiction_type: formData.addictionType,
        severity: formData.severity,
        frequency: formData.frequency,
        duration_years: formData.durationYears ? parseFloat(formData.durationYears) : undefined,
        daily_cost: formData.dailyCost ? parseFloat(formData.dailyCost) : undefined,
        previous_attempts: parseInt(formData.previousAttempts, 10) || 0,
        support_system: formData.supportSystem,
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

  if (loading) {
    return <LoadingSpinner />;
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <ScrollView style={styles.scrollView}>
        <View style={styles.header}>
          <Text style={styles.title}>Evaluación de Adicción</Text>
          <Text style={styles.subtitle}>
            Completa esta evaluación para obtener un plan personalizado
          </Text>
        </View>

        {!assessment ? (
          <AssessmentForm onSubmit={handleAssessment} />
        ) : (
          <AssessmentResults
            assessment={assessment}
            onNewAssessment={() => setAssessment(null)}
          />
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

