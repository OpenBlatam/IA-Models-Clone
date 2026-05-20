import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '@/contexts/theme-context';
import { formatPercentage } from '@/utils/numbers';
import type { QuotaInfo } from '@/types/api';

interface QuotaIndicatorProps {
  quota: QuotaInfo;
  showDetails?: boolean;
}

export function QuotaIndicator({ quota, showDetails = true }: QuotaIndicatorProps) {
  const { colors } = useTheme();

  const videosPercentage = (quota.videos_generated / quota.videos_limit) * 100;
  const storagePercentage = (quota.storage_used / quota.storage_limit) * 100;

  const getColor = (percentage: number) => {
    if (percentage >= 90) return colors.error;
    if (percentage >= 75) return colors.warning;
    return colors.success;
  };

  return (
    <View style={styles.container}>
      <View style={styles.section}>
        <View style={styles.header}>
          <Text style={[styles.label, { color: colors.text }]}>Videos</Text>
          <Text style={[styles.value, { color: colors.text }]}>
            {quota.videos_generated} / {quota.videos_limit}
          </Text>
        </View>
        <View style={[styles.progressBar, { backgroundColor: colors.border }]}>
          <View
            style={[
              styles.progressFill,
              {
                width: `${Math.min(videosPercentage, 100)}%`,
                backgroundColor: getColor(videosPercentage),
              },
            ]}
          />
        </View>
        {showDetails && (
          <Text style={[styles.percentage, { color: colors.textSecondary }]}>
            {formatPercentage(videosPercentage, 1)}
          </Text>
        )}
      </View>

      <View style={styles.section}>
        <View style={styles.header}>
          <Text style={[styles.label, { color: colors.text }]}>Storage</Text>
          <Text style={[styles.value, { color: colors.text }]}>
            {(quota.storage_used / 1024 / 1024).toFixed(2)} MB /{' '}
            {(quota.storage_limit / 1024 / 1024).toFixed(2)} MB
          </Text>
        </View>
        <View style={[styles.progressBar, { backgroundColor: colors.border }]}>
          <View
            style={[
              styles.progressFill,
              {
                width: `${Math.min(storagePercentage, 100)}%`,
                backgroundColor: getColor(storagePercentage),
              },
            ]}
          />
        </View>
        {showDetails && (
          <Text style={[styles.percentage, { color: colors.textSecondary }]}>
            {formatPercentage(storagePercentage, 1)}
          </Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: 16,
  },
  section: {
    marginBottom: 8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
  },
  value: {
    fontSize: 14,
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  percentage: {
    fontSize: 12,
    marginTop: 4,
  },
});

