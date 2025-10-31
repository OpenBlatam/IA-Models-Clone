import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  AccessibilityInfo,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

interface DashboardCardProps {
  title: string;
  subtitle: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  icon: string;
  onPress: () => void;
  disabled?: boolean;
}

export function DashboardCard({
  title,
  subtitle,
  status,
  icon,
  onPress,
  disabled = false,
}: DashboardCardProps): JSX.Element {
  const { theme } = useTheme();

  const statusColor = getStatusColor(status, theme);
  const styles = createStyles(theme, statusColor);

  const accessibilityLabel = `${title}, ${subtitle}, Status: ${status}`;

  return (
    <TouchableOpacity
      style={[styles.container, disabled && styles.disabled]}
      onPress={onPress}
      disabled={disabled}
      accessible={true}
      accessibilityLabel={accessibilityLabel}
      accessibilityRole="button"
      accessibilityState={{ disabled }}
    >
      <View style={styles.iconContainer}>
        <Ionicons name={icon as any} size={24} color={theme.colors.primary} />
      </View>
      
      <View style={styles.content}>
        <Text style={styles.title} numberOfLines={1}>
          {title}
        </Text>
        <Text style={styles.subtitle} numberOfLines={2}>
          {subtitle}
        </Text>
      </View>

      <View style={styles.statusContainer}>
        <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
        <Text style={styles.statusText}>
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </Text>
      </View>
    </TouchableOpacity>
  );
}

function getStatusColor(status: string, theme: any): string {
  switch (status) {
    case 'healthy':
      return theme.colors.success;
    case 'warning':
      return theme.colors.warning;
    case 'error':
      return theme.colors.error;
    case 'unknown':
      return theme.colors.textSecondary;
    default:
      return theme.colors.textSecondary;
  }
}

function createStyles(theme: any, statusColor: string) {
  return StyleSheet.create({
    container: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.md,
      borderLeftWidth: 4,
      borderLeftColor: statusColor,
      shadowColor: theme.colors.text,
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    disabled: {
      opacity: 0.5,
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
    content: {
      flex: 1,
      marginRight: theme.spacing.md,
    },
    title: {
      fontSize: theme.typography.h5.fontSize,
      fontWeight: theme.typography.h5.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    subtitle: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      lineHeight: theme.typography.caption.lineHeight,
    },
    statusContainer: {
      alignItems: 'center',
      minWidth: 60,
    },
    statusDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      marginBottom: theme.spacing.xs,
    },
    statusText: {
      fontSize: theme.typography.caption.fontSize,
      color: statusColor,
      fontWeight: '600',
      textAlign: 'center',
    },
  });
}
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  AccessibilityInfo,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

interface DashboardCardProps {
  title: string;
  subtitle: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  icon: string;
  onPress: () => void;
  disabled?: boolean;
}

export function DashboardCard({
  title,
  subtitle,
  status,
  icon,
  onPress,
  disabled = false,
}: DashboardCardProps): JSX.Element {
  const { theme } = useTheme();

  const statusColor = getStatusColor(status, theme);
  const styles = createStyles(theme, statusColor);

  const accessibilityLabel = `${title}, ${subtitle}, Status: ${status}`;

  return (
    <TouchableOpacity
      style={[styles.container, disabled && styles.disabled]}
      onPress={onPress}
      disabled={disabled}
      accessible={true}
      accessibilityLabel={accessibilityLabel}
      accessibilityRole="button"
      accessibilityState={{ disabled }}
    >
      <View style={styles.iconContainer}>
        <Ionicons name={icon as any} size={24} color={theme.colors.primary} />
      </View>
      
      <View style={styles.content}>
        <Text style={styles.title} numberOfLines={1}>
          {title}
        </Text>
        <Text style={styles.subtitle} numberOfLines={2}>
          {subtitle}
        </Text>
      </View>

      <View style={styles.statusContainer}>
        <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
        <Text style={styles.statusText}>
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </Text>
      </View>
    </TouchableOpacity>
  );
}

function getStatusColor(status: string, theme: any): string {
  switch (status) {
    case 'healthy':
      return theme.colors.success;
    case 'warning':
      return theme.colors.warning;
    case 'error':
      return theme.colors.error;
    case 'unknown':
      return theme.colors.textSecondary;
    default:
      return theme.colors.textSecondary;
  }
}

function createStyles(theme: any, statusColor: string) {
  return StyleSheet.create({
    container: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.md,
      borderLeftWidth: 4,
      borderLeftColor: statusColor,
      shadowColor: theme.colors.text,
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    disabled: {
      opacity: 0.5,
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
    content: {
      flex: 1,
      marginRight: theme.spacing.md,
    },
    title: {
      fontSize: theme.typography.h5.fontSize,
      fontWeight: theme.typography.h5.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    subtitle: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      lineHeight: theme.typography.caption.lineHeight,
    },
    statusContainer: {
      alignItems: 'center',
      minWidth: 60,
    },
    statusDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      marginBottom: theme.spacing.xs,
    },
    statusText: {
      fontSize: theme.typography.caption.fontSize,
      color: statusColor,
      fontWeight: '600',
      textAlign: 'center',
    },
  });
}


