import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Share,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRoute, useNavigation } from '@react-navigation/native';
import { RouteProp } from '@react-navigation/stack';
import { RootStackParamList, AnalysisResult } from '../types';
import ApiService from '../services/apiService';
import ScoreCard from '../components/ScoreCard';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorView from '../components/ErrorView';
import { getScoreColor, getScoreLabel } from '../utils/helpers';

type ReportScreenRouteProp = RouteProp<RootStackParamList, 'Report'>;

const ReportScreen: React.FC = () => {
  const route = useRoute<ReportScreenRouteProp>();
  const navigation = useNavigation();
  const { analysis } = route.params;

  const [report, setReport] = useState<AnalysisResult | null>(analysis);
  const [isLoading, setIsLoading] = useState(false);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    if (analysis?.id && !analysis.quality_scores) {
      loadReport();
    }
  }, []);

  const loadReport = async () => {
    if (!analysis?.id) return;

    try {
      setIsLoading(true);
      const result = await ApiService.getReport(analysis.id, 'json');
      setReport(result.data || result.report || result);
    } catch (error) {
      console.error('Error loading report:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const exportReport = async (format: 'pdf' | 'html') => {
    if (!analysis?.id) return;

    try {
      setExporting(true);
      const result = await ApiService.getReport(analysis.id, format);
      Alert.alert('Éxito', `Reporte exportado en formato ${format.toUpperCase()}`);
    } catch (error) {
      Alert.alert('Error', 'No se pudo exportar el reporte');
    } finally {
      setExporting(false);
    }
  };

  const shareReport = async () => {
    if (!report) return;

    try {
      const reportText = `
Análisis de Piel - Dermatology AI

Puntuación General: ${Math.round(report.quality_scores?.overall_score || 0)}
Tipo de Piel: ${report.skin_type || 'No detectado'}

Puntuaciones:
- Textura: ${Math.round(report.quality_scores?.texture_score || 0)}
- Hidratación: ${Math.round(report.quality_scores?.hydration_score || 0)}
- Elasticidad: ${Math.round(report.quality_scores?.elasticity_score || 0)}
- Pigmentación: ${Math.round(report.quality_scores?.pigmentation_score || 0)}

${report.conditions && report.conditions.length > 0
  ? `Condiciones detectadas: ${report.conditions.length}`
  : 'No se detectaron condiciones'}
      `.trim();

      await Share.share({
        message: reportText,
        title: 'Análisis de Piel',
      });
    } catch (error) {
      console.error('Error sharing report:', error);
    }
  };

  if (isLoading && !report) {
    return <LoadingSpinner message="Generando reporte..." />;
  }

  if (!report) {
    return (
      <ErrorView
        message="No se pudo cargar el reporte"
        onRetry={loadReport}
      />
    );
  }

  const scores = report.quality_scores || {};
  const overallScore = scores.overall_score || 0;
  const scoreColor = getScoreColor(overallScore);
  const scoreLabel = getScoreLabel(overallScore);

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Ionicons name="document-text" size={32} color="#6366f1" />
          <Text style={styles.headerTitle}>Reporte Detallado</Text>
          <Text style={styles.headerSubtitle}>
            Análisis completo de tu piel
          </Text>
        </View>
      </View>

      <View style={styles.content}>
        {/* Overall Score Card */}
        <View style={[styles.overallScoreCard, { borderColor: scoreColor }]}>
          <Text style={styles.overallScoreLabel}>Puntuación General</Text>
          <Text style={[styles.overallScore, { color: scoreColor }]}>
            {Math.round(overallScore)}
          </Text>
          <Text style={[styles.overallScoreText, { color: scoreColor }]}>
            {scoreLabel}
          </Text>
        </View>

        {/* Scores Grid */}
        {scores.overall_score && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Puntuaciones Detalladas</Text>
            <View style={styles.scoresGrid}>
              <ScoreCard
                label="Textura"
                value={scores.texture_score}
                color="#8b5cf6"
              />
              <ScoreCard
                label="Hidratación"
                value={scores.hydration_score}
                color="#ec4899"
              />
              <ScoreCard
                label="Elasticidad"
                value={scores.elasticity_score}
                color="#f59e0b"
              />
              <ScoreCard
                label="Pigmentación"
                value={scores.pigmentation_score}
                color="#10b981"
              />
            </View>
          </View>
        )}

        {/* Summary */}
        {report.summary && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Resumen</Text>
            <View style={styles.summaryCard}>
              <Text style={styles.summaryText}>{report.summary}</Text>
            </View>
          </View>
        )}

        {/* Conditions */}
        {report.conditions && report.conditions.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Condiciones Detectadas</Text>
            {report.conditions.map((condition, index) => (
              <View key={index} style={styles.conditionCard}>
                <Ionicons name="warning" size={24} color="#f59e0b" />
                <View style={styles.conditionContent}>
                  <Text style={styles.conditionName}>{condition.name}</Text>
                  {condition.severity && (
                    <Text style={styles.conditionSeverity}>
                      Severidad: {condition.severity}
                    </Text>
                  )}
                  {condition.confidence && (
                    <Text style={styles.conditionConfidence}>
                      Confianza: {Math.round(condition.confidence)}%
                    </Text>
                  )}
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Recommendations */}
        {report.recommendations_priority && report.recommendations_priority.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Recomendaciones Prioritarias</Text>
            {report.recommendations_priority.map((rec, index) => (
              <View key={index} style={styles.recommendationItem}>
                <Ionicons name="checkmark-circle" size={20} color="#10b981" />
                <Text style={styles.recommendationText}>{rec}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Actions */}
        <View style={styles.actionsSection}>
          <TouchableOpacity
            style={styles.shareButton}
            onPress={shareReport}
            activeOpacity={0.8}
          >
            <Ionicons name="share-social" size={24} color="#6366f1" />
            <Text style={styles.shareButtonText}>Compartir Reporte</Text>
          </TouchableOpacity>

          <View style={styles.exportButtons}>
            <TouchableOpacity
              style={styles.exportButton}
              onPress={() => exportReport('pdf')}
              disabled={exporting}
              activeOpacity={0.8}
            >
              {exporting ? (
                <ActivityIndicator color="#ef4444" />
              ) : (
                <>
                  <Ionicons name="document" size={24} color="#ef4444" />
                  <Text style={styles.exportButtonText}>PDF</Text>
                </>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.exportButton}
              onPress={() => exportReport('html')}
              disabled={exporting}
              activeOpacity={0.8}
            >
              {exporting ? (
                <ActivityIndicator color="#6366f1" />
              ) : (
                <>
                  <Ionicons name="code" size={24} color="#6366f1" />
                  <Text style={styles.exportButtonText}>HTML</Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#fff',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  headerContent: {
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 12,
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#6b7280',
  },
  content: {
    padding: 20,
  },
  overallScoreCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginBottom: 24,
    borderWidth: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 8,
  },
  overallScoreLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 8,
    textTransform: 'uppercase',
    fontWeight: '600',
  },
  overallScore: {
    fontSize: 64,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  overallScoreText: {
    fontSize: 18,
    fontWeight: '600',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
  },
  scoresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  summaryText: {
    fontSize: 16,
    color: '#4b5563',
    lineHeight: 24,
  },
  conditionCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  conditionContent: {
    flex: 1,
    marginLeft: 12,
  },
  conditionName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  conditionSeverity: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 2,
  },
  conditionConfidence: {
    fontSize: 12,
    color: '#9ca3af',
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#4b5563',
    marginLeft: 12,
    lineHeight: 20,
  },
  actionsSection: {
    marginTop: 8,
  },
  shareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#6366f1',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  shareButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  exportButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  exportButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginHorizontal: 4,
    borderWidth: 2,
    borderColor: '#e5e7eb',
  },
  exportButtonText: {
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
    color: '#1f2937',
  },
});

export default ReportScreen;

