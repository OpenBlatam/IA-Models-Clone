import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

interface ScoreCardProps {
  label: string;
  value?: number;
  color?: string;
}

const ScoreCard: React.FC<ScoreCardProps> = ({ 
  label, 
  value = 0, 
  color = '#6366f1' 
}) => {
  const percentage = Math.round(value);

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={[color, `${color}80`]}
        style={styles.gradient}
      >
        <Text style={styles.score}>{percentage}</Text>
        <View style={styles.progressBar}>
          <View
            style={[
              styles.progressFill,
              { width: `${percentage}%`, backgroundColor: '#fff' },
            ]}
          />
        </View>
        <Text style={styles.label}>{label}</Text>
      </LinearGradient>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '48%',
    marginBottom: 12,
    borderRadius: 12,
    overflow: 'hidden',
  },
  gradient: {
    padding: 16,
    alignItems: 'center',
  },
  score: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  progressBar: {
    width: '100%',
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: 2,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
  },
  label: {
    fontSize: 12,
    color: '#fff',
    fontWeight: '600',
    textTransform: 'uppercase',
  },
});

export default ScoreCard;

