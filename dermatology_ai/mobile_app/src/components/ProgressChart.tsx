import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import { QualityScores } from '../types';

interface ProgressChartProps {
  data: QualityScores[];
  labels: string[];
}

const ProgressChart: React.FC<ProgressChartProps> = ({ data, labels }) => {
  if (!data || data.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyText}>No hay datos de progreso disponibles</Text>
      </View>
    );
  }

  const screenWidth = Dimensions.get('window').width;
  const chartData = {
    labels: labels.length > 0 ? labels : data.map((_, i) => `Día ${i + 1}`),
    datasets: [
      {
        data: data.map((score) => score.overall_score || 0),
        color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
        strokeWidth: 2,
      },
    ],
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Progreso de Puntuación</Text>
      <LineChart
        data={chartData}
        width={screenWidth - 40}
        height={220}
        chartConfig={{
          backgroundColor: '#ffffff',
          backgroundGradientFrom: '#ffffff',
          backgroundGradientTo: '#ffffff',
          decimalPlaces: 0,
          color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
          labelColor: (opacity = 1) => `rgba(107, 114, 128, ${opacity})`,
          style: {
            borderRadius: 16,
          },
          propsForDots: {
            r: '6',
            strokeWidth: '2',
            stroke: '#6366f1',
          },
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
  emptyContainer: {
    padding: 20,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
  },
});

export default ProgressChart;

