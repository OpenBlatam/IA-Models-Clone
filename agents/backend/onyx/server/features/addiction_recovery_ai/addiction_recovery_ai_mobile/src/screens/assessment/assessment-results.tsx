import React from 'react';
import { View, Text } from 'react-native';
import { Button } from '@/components';
import { useColors } from '@/theme/colors';
import type { AssessmentResponse } from '@/types';
import { useAssessmentStyles } from './assessment-screen.styles';

interface AssessmentResultsProps {
  assessment: AssessmentResponse;
  onNewAssessment: () => void;
}

export function AssessmentResults({
  assessment,
  onNewAssessment,
}: AssessmentResultsProps): JSX.Element {
  const colors = useColors();
  const styles = useAssessmentStyles(colors);

  const riskLevelColor =
    assessment.risk_level === 'low'
      ? colors.success
      : assessment.risk_level === 'moderate'
      ? colors.warning
      : assessment.risk_level === 'high'
      ? colors.error
      : colors.textSecondary;

  const riskLevelText =
    assessment.risk_level === 'low'
      ? 'Bajo'
      : assessment.risk_level === 'moderate'
      ? 'Moderado'
      : assessment.risk_level === 'high'
      ? 'Alto'
      : 'Crítico';

  return (
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
        <Text style={[styles.resultValue, { color: riskLevelColor }]}>
          {riskLevelText}
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
        onPress={onNewAssessment}
        variant="outline"
        style={styles.button}
      />
    </View>
  );
}

