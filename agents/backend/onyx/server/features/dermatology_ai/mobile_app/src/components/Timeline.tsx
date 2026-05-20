import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { formatDate } from '../utils/helpers';

interface TimelineItem {
  id: string;
  date: string;
  title: string;
  description?: string;
  score?: number;
  type: 'analysis' | 'recommendation' | 'goal';
}

interface TimelineProps {
  items: TimelineItem[];
}

const Timeline: React.FC<TimelineProps> = ({ items }) => {
  const getIcon = (type: TimelineItem['type']) => {
    switch (type) {
      case 'analysis':
        return 'analytics';
      case 'recommendation':
        return 'sparkles';
      case 'goal':
        return 'flag';
      default:
        return 'circle';
    }
  };

  const getColor = (type: TimelineItem['type']) => {
    switch (type) {
      case 'analysis':
        return '#6366f1';
      case 'recommendation':
        return '#8b5cf6';
      case 'goal':
        return '#10b981';
      default:
        return '#9ca3af';
    }
  };

  return (
    <View style={styles.container}>
      {items.map((item, index) => (
        <View key={item.id} style={styles.item}>
          <View style={styles.leftSection}>
            <View
              style={[
                styles.iconContainer,
                { backgroundColor: `${getColor(item.type)}20` },
              ]}
            >
              <Ionicons
                name={getIcon(item.type)}
                size={24}
                color={getColor(item.type)}
              />
            </View>
            {index < items.length - 1 && (
              <View style={[styles.line, { backgroundColor: getColor(item.type) }]} />
            )}
          </View>
          <View style={styles.content}>
            <Text style={styles.date}>{formatDate(item.date)}</Text>
            <Text style={styles.title}>{item.title}</Text>
            {item.description && (
              <Text style={styles.description}>{item.description}</Text>
            )}
            {item.score !== undefined && (
              <View style={styles.scoreContainer}>
                <Text style={styles.score}>{Math.round(item.score)}</Text>
                <Text style={styles.scoreLabel}>Puntuación</Text>
              </View>
            )}
          </View>
        </View>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  item: {
    flexDirection: 'row',
    marginBottom: 24,
  },
  leftSection: {
    alignItems: 'center',
    marginRight: 16,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1,
  },
  line: {
    width: 2,
    flex: 1,
    marginTop: 8,
    opacity: 0.3,
  },
  content: {
    flex: 1,
    paddingTop: 4,
  },
  date: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  description: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
    marginBottom: 8,
  },
  scoreContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  score: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#6366f1',
    marginRight: 8,
  },
  scoreLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
});

export default Timeline;

