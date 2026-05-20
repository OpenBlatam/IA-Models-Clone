import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
  onPress?: () => void;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  icon,
  color,
  onPress,
  trend,
}) => {
  const content = (
    <LinearGradient
      colors={[color, `${color}80`]}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <View style={styles.iconContainer}>
        <Ionicons name={icon} size={32} color="#fff" />
      </View>
      <Text style={styles.value}>{value}</Text>
      <Text style={styles.title}>{title}</Text>
      {trend && (
        <View style={styles.trendContainer}>
          <Ionicons
            name={trend.isPositive ? 'trending-up' : 'trending-down'}
            size={16}
            color="#fff"
          />
          <Text style={styles.trendText}>
            {trend.isPositive ? '+' : ''}{trend.value}%
          </Text>
        </View>
      )}
    </LinearGradient>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.8}>
        {content}
      </TouchableOpacity>
    );
  }

  return content;
};

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    padding: 20,
    minHeight: 140,
    justifyContent: 'space-between',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  iconContainer: {
    marginBottom: 12,
  },
  value: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  title: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
    fontWeight: '500',
  },
  trendContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  trendText: {
    fontSize: 12,
    color: '#fff',
    marginLeft: 4,
    fontWeight: '600',
  },
});

export default StatsCard;

