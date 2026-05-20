import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

export function SystemStatusCard(): JSX.Element {
  const { theme } = useTheme();
  const styles = createStyles(theme);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="shield-checkmark-outline" size={24} color={theme.colors.success} />
        <Text style={styles.title}>System Status</Text>
      </View>
      
      <View style={styles.statusGrid}>
        <View style={styles.statusItem}>
          <View style={[styles.statusIndicator, { backgroundColor: theme.colors.success }]} />
          <Text style={styles.statusLabel}>Overall</Text>
          <Text style={styles.statusValue}>Healthy</Text>
        </View>
        
        <View style={styles.statusItem}>
          <View style={[styles.statusIndicator, { backgroundColor: theme.colors.success }]} />
          <Text style={styles.statusLabel}>CPU</Text>
          <Text style={styles.statusValue}>45%</Text>
        </View>
        
        <View style={styles.statusItem}>
          <View style={[styles.statusIndicator, { backgroundColor: theme.colors.warning }]} />
          <Text style={styles.statusLabel}>Memory</Text>
          <Text style={styles.statusValue}>78%</Text>
        </View>
        
        <View style={styles.statusItem}>
          <View style={[styles.statusIndicator, { backgroundColor: theme.colors.success }]} />
          <Text style={styles.statusLabel}>Network</Text>
          <Text style={styles.statusValue}>Stable</Text>
        </View>
      </View>
      
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Last updated: {new Date().toLocaleTimeString()}
        </Text>
      </View>
    </View>
  );
}

function createStyles(theme: any) {
  return StyleSheet.create({
    container: {
      backgroundColor: theme.colors.surface,
      margin: theme.spacing.lg,
      borderRadius: theme.borderRadius.lg,
      padding: theme.spacing.lg,
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
      marginBottom: theme.spacing.lg,
    },
    title: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginLeft: theme.spacing.sm,
    },
    statusGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginBottom: theme.spacing.lg,
    },
    statusItem: {
      alignItems: 'center',
      flex: 1,
    },
    statusIndicator: {
      width: 12,
      height: 12,
      borderRadius: 6,
      marginBottom: theme.spacing.xs,
    },
    statusLabel: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.xs,
    },
    statusValue: {
      fontSize: theme.typography.body.fontSize,
      fontWeight: '600',
      color: theme.colors.text,
    },
    footer: {
      borderTopWidth: 1,
      borderTopColor: theme.colors.border,
      paddingTop: theme.spacing.md,
    },
    footerText: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      textAlign: 'center',
    },
  });
}
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

export function SystemStatusCard(): JSX.Element {
  const { theme } = useTheme();
  const styles = createStyles(theme);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="shield-checkmark-outline" size={24} color={theme.colors.success} />
        <Text style={styles.title}>System Status</Text>
      </View>
      
      <View style={styles.statusGrid}>
        <View style={styles.statusItem}>
          <View style={[styles.statusIndicator, { backgroundColor: theme.colors.success }]} />
          <Text style={styles.statusLabel}>Overall</Text>
          <Text style={styles.statusValue}>Healthy</Text>
        </View>
        
        <View style={styles.statusItem}>
          <View style={[styles.statusIndicator, { backgroundColor: theme.colors.success }]} />
          <Text style={styles.statusLabel}>CPU</Text>
          <Text style={styles.statusValue}>45%</Text>
        </View>
        
        <View style={styles.statusItem}>
          <View style={[styles.statusIndicator, { backgroundColor: theme.colors.warning }]} />
          <Text style={styles.statusLabel}>Memory</Text>
          <Text style={styles.statusValue}>78%</Text>
        </View>
        
        <View style={styles.statusItem}>
          <View style={[styles.statusIndicator, { backgroundColor: theme.colors.success }]} />
          <Text style={styles.statusLabel}>Network</Text>
          <Text style={styles.statusValue}>Stable</Text>
        </View>
      </View>
      
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Last updated: {new Date().toLocaleTimeString()}
        </Text>
      </View>
    </View>
  );
}

function createStyles(theme: any) {
  return StyleSheet.create({
    container: {
      backgroundColor: theme.colors.surface,
      margin: theme.spacing.lg,
      borderRadius: theme.borderRadius.lg,
      padding: theme.spacing.lg,
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
      marginBottom: theme.spacing.lg,
    },
    title: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginLeft: theme.spacing.sm,
    },
    statusGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginBottom: theme.spacing.lg,
    },
    statusItem: {
      alignItems: 'center',
      flex: 1,
    },
    statusIndicator: {
      width: 12,
      height: 12,
      borderRadius: 6,
      marginBottom: theme.spacing.xs,
    },
    statusLabel: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.xs,
    },
    statusValue: {
      fontSize: theme.typography.body.fontSize,
      fontWeight: '600',
      color: theme.colors.text,
    },
    footer: {
      borderTopWidth: 1,
      borderTopColor: theme.colors.border,
      paddingTop: theme.spacing.md,
    },
    footerText: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      textAlign: 'center',
    },
  });
}


