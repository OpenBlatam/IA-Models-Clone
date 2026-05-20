import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

interface Module {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'error' | 'maintenance';
  type: string;
  version: string;
  icon: string;
  metrics: {
    cpu: number;
    memory: number;
    uptime: number;
  };
}

interface ModuleCardProps {
  module: Module;
  onPress: () => void;
}

export function ModuleCard({ module, onPress }: ModuleCardProps): JSX.Element {
  const { theme } = useTheme();
  const styles = createStyles(theme);

  const statusColor = getStatusColor(module.status, theme);
  const statusText = getStatusText(module.status);

  return (
    <TouchableOpacity
      style={[styles.container, { borderLeftColor: statusColor }]}
      onPress={onPress}
      accessible={true}
      accessibilityLabel={`${module.name}, Status: ${statusText}, Type: ${module.type}`}
      accessibilityRole="button"
    >
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.iconContainer}>
          <Ionicons name={module.icon as any} size={24} color={theme.colors.primary} />
        </View>
        
        <View style={styles.headerContent}>
          <Text style={styles.name} numberOfLines={1}>
            {module.name}
          </Text>
          <Text style={styles.type}>{module.type}</Text>
        </View>

        <View style={styles.statusContainer}>
          <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
          <Text style={[styles.statusText, { color: statusColor }]}>
            {statusText}
          </Text>
        </View>
      </View>

      {/* Description */}
      <Text style={styles.description} numberOfLines={2}>
        {module.description}
      </Text>

      {/* Metrics */}
      <View style={styles.metricsContainer}>
        <View style={styles.metricItem}>
          <Ionicons name="hardware-chip-outline" size={16} color={theme.colors.textSecondary} />
          <Text style={styles.metricValue}>{module.metrics.cpu}%</Text>
          <Text style={styles.metricLabel}>CPU</Text>
        </View>
        
        <View style={styles.metricItem}>
          <Ionicons name="memory-outline" size={16} color={theme.colors.textSecondary} />
          <Text style={styles.metricValue}>{module.metrics.memory}%</Text>
          <Text style={styles.metricLabel}>Memory</Text>
        </View>
        
        <View style={styles.metricItem}>
          <Ionicons name="time-outline" size={16} color={theme.colors.textSecondary} />
          <Text style={styles.metricValue}>{module.metrics.uptime}%</Text>
          <Text style={styles.metricLabel}>Uptime</Text>
        </View>
      </View>

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.version}>v{module.version}</Text>
        <View style={styles.actions}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => console.log(`Configure ${module.name}`)}
            accessible={true}
            accessibilityLabel={`Configure ${module.name}`}
            accessibilityRole="button"
          >
            <Ionicons name="settings-outline" size={16} color={theme.colors.primary} />
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => console.log(`View logs for ${module.name}`)}
            accessible={true}
            accessibilityLabel={`View logs for ${module.name}`}
            accessibilityRole="button"
          >
            <Ionicons name="document-text-outline" size={16} color={theme.colors.primary} />
          </TouchableOpacity>
        </View>
      </View>
    </TouchableOpacity>
  );
}

function getStatusColor(status: string, theme: any): string {
  switch (status) {
    case 'active':
      return theme.colors.success;
    case 'inactive':
      return theme.colors.textSecondary;
    case 'error':
      return theme.colors.error;
    case 'maintenance':
      return theme.colors.warning;
    default:
      return theme.colors.textSecondary;
  }
}

function getStatusText(status: string): string {
  switch (status) {
    case 'active':
      return 'Active';
    case 'inactive':
      return 'Inactive';
    case 'error':
      return 'Error';
    case 'maintenance':
      return 'Maintenance';
    default:
      return 'Unknown';
  }
}

function createStyles(theme: any) {
  return StyleSheet.create({
    container: {
      backgroundColor: theme.colors.surface,
      borderRadius: theme.borderRadius.lg,
      padding: theme.spacing.lg,
      marginBottom: theme.spacing.md,
      borderLeftWidth: 4,
      shadowColor: theme.colors.text,
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    header: {
      flexDirection: 'row',
      alignItems: 'center',
      marginBottom: theme.spacing.md,
    },
    iconContainer: {
      width: 48,
      height: 48,
      borderRadius: 24,
      backgroundColor: `${theme.colors.primary}15`,
      justifyContent: 'center',
      alignItems: 'center',
      marginRight: theme.spacing.md,
    },
    headerContent: {
      flex: 1,
      marginRight: theme.spacing.md,
    },
    name: {
      fontSize: theme.typography.h5.fontSize,
      fontWeight: theme.typography.h5.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    type: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      fontWeight: '500',
    },
    statusContainer: {
      alignItems: 'center',
      minWidth: 80,
    },
    statusDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      marginBottom: theme.spacing.xs,
    },
    statusText: {
      fontSize: theme.typography.caption.fontSize,
      fontWeight: '600',
      textAlign: 'center',
    },
    description: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.textSecondary,
      lineHeight: theme.typography.body.lineHeight,
      marginBottom: theme.spacing.lg,
    },
    metricsContainer: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginBottom: theme.spacing.lg,
      paddingVertical: theme.spacing.md,
      borderTopWidth: 1,
      borderTopColor: theme.colors.border,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    metricItem: {
      alignItems: 'center',
      flex: 1,
    },
    metricValue: {
      fontSize: theme.typography.h6.fontSize,
      fontWeight: theme.typography.h6.fontWeight,
      color: theme.colors.text,
      marginTop: theme.spacing.xs,
      marginBottom: theme.spacing.xs,
    },
    metricLabel: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      textAlign: 'center',
    },
    footer: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    version: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      fontWeight: '500',
    },
    actions: {
      flexDirection: 'row',
      gap: theme.spacing.sm,
    },
    actionButton: {
      width: 32,
      height: 32,
      borderRadius: 16,
      backgroundColor: `${theme.colors.primary}15`,
      justifyContent: 'center',
      alignItems: 'center',
    },
  });
}
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

interface Module {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'error' | 'maintenance';
  type: string;
  version: string;
  icon: string;
  metrics: {
    cpu: number;
    memory: number;
    uptime: number;
  };
}

interface ModuleCardProps {
  module: Module;
  onPress: () => void;
}

export function ModuleCard({ module, onPress }: ModuleCardProps): JSX.Element {
  const { theme } = useTheme();
  const styles = createStyles(theme);

  const statusColor = getStatusColor(module.status, theme);
  const statusText = getStatusText(module.status);

  return (
    <TouchableOpacity
      style={[styles.container, { borderLeftColor: statusColor }]}
      onPress={onPress}
      accessible={true}
      accessibilityLabel={`${module.name}, Status: ${statusText}, Type: ${module.type}`}
      accessibilityRole="button"
    >
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.iconContainer}>
          <Ionicons name={module.icon as any} size={24} color={theme.colors.primary} />
        </View>
        
        <View style={styles.headerContent}>
          <Text style={styles.name} numberOfLines={1}>
            {module.name}
          </Text>
          <Text style={styles.type}>{module.type}</Text>
        </View>

        <View style={styles.statusContainer}>
          <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
          <Text style={[styles.statusText, { color: statusColor }]}>
            {statusText}
          </Text>
        </View>
      </View>

      {/* Description */}
      <Text style={styles.description} numberOfLines={2}>
        {module.description}
      </Text>

      {/* Metrics */}
      <View style={styles.metricsContainer}>
        <View style={styles.metricItem}>
          <Ionicons name="hardware-chip-outline" size={16} color={theme.colors.textSecondary} />
          <Text style={styles.metricValue}>{module.metrics.cpu}%</Text>
          <Text style={styles.metricLabel}>CPU</Text>
        </View>
        
        <View style={styles.metricItem}>
          <Ionicons name="memory-outline" size={16} color={theme.colors.textSecondary} />
          <Text style={styles.metricValue}>{module.metrics.memory}%</Text>
          <Text style={styles.metricLabel}>Memory</Text>
        </View>
        
        <View style={styles.metricItem}>
          <Ionicons name="time-outline" size={16} color={theme.colors.textSecondary} />
          <Text style={styles.metricValue}>{module.metrics.uptime}%</Text>
          <Text style={styles.metricLabel}>Uptime</Text>
        </View>
      </View>

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.version}>v{module.version}</Text>
        <View style={styles.actions}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => console.log(`Configure ${module.name}`)}
            accessible={true}
            accessibilityLabel={`Configure ${module.name}`}
            accessibilityRole="button"
          >
            <Ionicons name="settings-outline" size={16} color={theme.colors.primary} />
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => console.log(`View logs for ${module.name}`)}
            accessible={true}
            accessibilityLabel={`View logs for ${module.name}`}
            accessibilityRole="button"
          >
            <Ionicons name="document-text-outline" size={16} color={theme.colors.primary} />
          </TouchableOpacity>
        </View>
      </View>
    </TouchableOpacity>
  );
}

function getStatusColor(status: string, theme: any): string {
  switch (status) {
    case 'active':
      return theme.colors.success;
    case 'inactive':
      return theme.colors.textSecondary;
    case 'error':
      return theme.colors.error;
    case 'maintenance':
      return theme.colors.warning;
    default:
      return theme.colors.textSecondary;
  }
}

function getStatusText(status: string): string {
  switch (status) {
    case 'active':
      return 'Active';
    case 'inactive':
      return 'Inactive';
    case 'error':
      return 'Error';
    case 'maintenance':
      return 'Maintenance';
    default:
      return 'Unknown';
  }
}

function createStyles(theme: any) {
  return StyleSheet.create({
    container: {
      backgroundColor: theme.colors.surface,
      borderRadius: theme.borderRadius.lg,
      padding: theme.spacing.lg,
      marginBottom: theme.spacing.md,
      borderLeftWidth: 4,
      shadowColor: theme.colors.text,
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    header: {
      flexDirection: 'row',
      alignItems: 'center',
      marginBottom: theme.spacing.md,
    },
    iconContainer: {
      width: 48,
      height: 48,
      borderRadius: 24,
      backgroundColor: `${theme.colors.primary}15`,
      justifyContent: 'center',
      alignItems: 'center',
      marginRight: theme.spacing.md,
    },
    headerContent: {
      flex: 1,
      marginRight: theme.spacing.md,
    },
    name: {
      fontSize: theme.typography.h5.fontSize,
      fontWeight: theme.typography.h5.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    type: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      fontWeight: '500',
    },
    statusContainer: {
      alignItems: 'center',
      minWidth: 80,
    },
    statusDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      marginBottom: theme.spacing.xs,
    },
    statusText: {
      fontSize: theme.typography.caption.fontSize,
      fontWeight: '600',
      textAlign: 'center',
    },
    description: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.textSecondary,
      lineHeight: theme.typography.body.lineHeight,
      marginBottom: theme.spacing.lg,
    },
    metricsContainer: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginBottom: theme.spacing.lg,
      paddingVertical: theme.spacing.md,
      borderTopWidth: 1,
      borderTopColor: theme.colors.border,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    metricItem: {
      alignItems: 'center',
      flex: 1,
    },
    metricValue: {
      fontSize: theme.typography.h6.fontSize,
      fontWeight: theme.typography.h6.fontWeight,
      color: theme.colors.text,
      marginTop: theme.spacing.xs,
      marginBottom: theme.spacing.xs,
    },
    metricLabel: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      textAlign: 'center',
    },
    footer: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    version: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      fontWeight: '500',
    },
    actions: {
      flexDirection: 'row',
      gap: theme.spacing.sm,
    },
    actionButton: {
      width: 32,
      height: 32,
      borderRadius: 16,
      backgroundColor: `${theme.colors.primary}15`,
      justifyContent: 'center',
      alignItems: 'center',
    },
  });
}


