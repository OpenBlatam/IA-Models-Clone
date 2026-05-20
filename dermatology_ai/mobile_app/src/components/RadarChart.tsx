import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { RadarChart as RNRadarChart } from 'react-native-chart-kit';
import { AnalysisResult } from '../types';

interface RadarChartProps {
  data: AnalysisResult | null;
}

const RadarChart: React.FC<RadarChartProps> = ({ data }) => {
  if (!data || !data.quality_scores) {
    return null;
  }

  const scores = data.quality_scores;
  const chartData = {
    labels: [
      'Textura',
      'Hidratación',
      'Elasticidad',
      'Pigmentación',
      'Poros',
      'Arrugas',
    ],
    datasets: [
      {
        data: [
          scores.texture_score || 0,
          scores.hydration_score || 0,
          scores.elasticity_score || 0,
          scores.pigmentation_score || 0,
          scores.pore_size_score || 0,
          scores.wrinkles_score || 0,
        ],
        color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
        strokeWidth: 2,
      },
    ],
  };

  const screenWidth = Dimensions.get('window').width;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Análisis Visual</Text>
      <RNRadarChart
        data={chartData}
        width={screenWidth - 40}
        height={220}
        chartConfig={{
          backgroundColor: '#ffffff',
          backgroundGradientFrom: '#ffffff',
          backgroundGradientTo: '#ffffff',
          color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
          strokeWidth: 2,
        }}
        bezier
        style={styles.chart}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  chart: {
    borderRadius: 12,
  },
});

export default RadarChart;

