import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { AnalysisResult } from '../types';
import ScoreCard from './ScoreCard';
import { getScoreColor, getScoreLabel } from '../utils/helpers';

interface ComparisonViewProps {
  analysis1: AnalysisResult;
  analysis2: AnalysisResult;
  label1?: string;
  label2?: string;
}

const ComparisonView: React.FC<ComparisonViewProps> = ({
  analysis1,
  analysis2,
  label1 = 'Análisis 1',
  label2 = 'Análisis 2',
}) => {
  const scores1 = analysis1.quality_scores || {};
  const scores2 = analysis2.quality_scores || {};

  const scoreItems = [
    { key: 'overall_score', label: 'General' },
    { key: 'texture_score', label: 'Textura' },
    { key: 'hydration_score', label: 'Hidratación' },
    { key: 'elasticity_score', label: 'Elasticidad' },
    { key: 'pigmentation_score', label: 'Pigmentación' },
    { key: 'pore_size_score', label: 'Poros' },
    { key: 'wrinkles_score', label: 'Arrugas' },
  ];

  const getDifference = (score1: number = 0, score2: number = 0): number => {
    return score2 - score1;
  };

  const getDifferenceColor = (difference: number): string => {
    if (difference > 0) return '#10b981'; // Green - improvement
    if (difference < 0) return '#ef4444'; // Red - decline
    return '#6b7280'; // Gray - no change
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Comparación de Análisis</Text>
      </View>

      <View style={styles.content}>
        {/* Overall Comparison */}
        <View style={styles.overallComparison}>
          <View style={styles.comparisonCard}>
            <Text style={styles.comparisonLabel}>{label1}</Text>
            <Text style={[styles.comparisonScore, { color: getScoreColor(scores1.overall_score || 0) }]}>
              {Math.round(scores1.overall_score || 0)}
            </Text>
            <Text style={styles.comparisonText}>{getScoreLabel(scores1.overall_score || 0)}</Text>
          </View>

          <View style={styles.arrowContainer}>
            <Ionicons name="arrow-forward" size={32} color="#6366f1" />
            <Text style={styles.differenceText}>
              {getDifference(scores1.overall_score, scores2.overall_score) > 0 ? '+' : ''}
              {Math.round(getDifference(scores1.overall_score, scores2.overall_score))}
            </Text>
          </View>

          <View style={styles.comparisonCard}>
            <Text style={styles.comparisonLabel}>{label2}</Text>
            <Text style={[styles.comparisonScore, { color: getScoreColor(scores2.overall_score || 0) }]}>
              {Math.round(scores2.overall_score || 0)}
            </Text>
            <Text style={styles.comparisonText}>{getScoreLabel(scores2.overall_score || 0)}</Text>
          </View>
        </View>

        {/* Detailed Scores */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Comparación Detallada</Text>
          {scoreItems.map((item) => {
            const score1 = (scores1 as any)[item.key] || 0;
            const score2 = (scores2 as any)[item.key] || 0;
            const difference = getDifference(score1, score2);
            const diffColor = getDifferenceColor(difference);

            return (
              <View key={item.key} style={styles.scoreComparisonRow}>
                <View style={styles.scoreInfo}>
                  <Text style={styles.scoreLabel}>{item.label}</Text>
                  <View style={styles.scoreValues}>
                    <Text style={styles.scoreValue}>{Math.round(score1)}</Text>
                    <Ionicons name="arrow-forward" size={16} color="#9ca3af" />
                    <Text style={styles.scoreValue}>{Math.round(score2)}</Text>
                  </View>
                </View>
                <View style={[styles.differenceBadge, { backgroundColor: `${diffColor}20` }]}>
                  <Ionicons
                    name={difference > 0 ? 'trending-up' : difference < 0 ? 'trending-down' : 'remove'}
                    size={16}
                    color={diffColor}
                  />
                  <Text style={[styles.differenceBadgeText, { color: diffColor }]}>
                    {difference > 0 ? '+' : ''}{Math.round(difference)}
                  </Text>
                </View>
              </View>
            );
          })}
        </View>

        {/* Conditions Comparison */}
        {(analysis1.conditions || analysis2.conditions) && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Condiciones</Text>
            <View style={styles.conditionsRow}>
              <View style={styles.conditionsColumn}>
                <Text style={styles.conditionsLabel}>{label1}</Text>
                <Text style={styles.conditionsCount}>
                  {analysis1.conditions?.length || 0} condición(es)
                </Text>
              </View>
              <View style={styles.conditionsColumn}>
                <Text style={styles.conditionsLabel}>{label2}</Text>
                <Text style={styles.conditionsCount}>
                  {analysis2.conditions?.length || 0} condición(es)
                </Text>
              </View>
            </View>
          </View>
        )}
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
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  content: {
    padding: 20,
  },
  overallComparison: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  comparisonCard: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  comparisonLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 8,
    fontWeight: '600',
  },
  comparisonScore: {
    fontSize: 36,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  comparisonText: {
    fontSize: 12,
    color: '#6b7280',
  },
  arrowContainer: {
    alignItems: 'center',
    marginHorizontal: 12,
  },
  differenceText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6366f1',
    marginTop: 4,
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
  scoreComparisonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  scoreInfo: {
    flex: 1,
  },
  scoreLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  scoreValues: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  scoreValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#6366f1',
    marginHorizontal: 8,
  },
  differenceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  differenceBadgeText: {
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 4,
  },
  conditionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  conditionsColumn: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 4,
    alignItems: 'center',
  },
  conditionsLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 8,
  },
  conditionsCount: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
  },
});

export default ComparisonView;

