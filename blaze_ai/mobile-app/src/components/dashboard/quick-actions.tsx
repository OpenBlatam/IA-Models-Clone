import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

interface QuickActionProps {
  title: string;
  icon: string;
  onPress: () => void;
  color?: string;
}

export function QuickActions(): JSX.Element {
  const { theme } = useTheme();
  const styles = createStyles(theme);

  const actions: QuickActionProps[] = [
    {
      title: 'Start Task',
      icon: 'play-outline',
      onPress: () => console.log('Start Task'),
      color: theme.colors.success,
    },
    {
      title: 'Stop Task',
      icon: 'stop-outline',
      onPress: () => console.log('Stop Task'),
      color: theme.colors.error,
    },
    {
      title: 'Restart',
      icon: 'refresh-outline',
      onPress: () => console.log('Restart'),
      color: theme.colors.warning,
    },
    {
      title: 'Settings',
      icon: 'settings-outline',
      onPress: () => console.log('Settings'),
      color: theme.colors.info,
    },
  ];

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Quick Actions</Text>
      <View style={styles.actionsGrid}>
        {actions.map((action, index) => (
          <QuickAction
            key={index}
            title={action.title}
            icon={action.icon}
            onPress={action.onPress}
            color={action.color}
          />
        ))}
      </View>
    </View>
  );
}

function QuickAction({ title, icon, onPress, color }: QuickActionProps): JSX.Element {
  const { theme } = useTheme();
  const styles = createActionStyles(theme, color);

  return (
    <TouchableOpacity
      style={styles.actionButton}
      onPress={onPress}
      accessible={true}
      accessibilityLabel={title}
      accessibilityRole="button"
    >
      <View style={styles.iconContainer}>
        <Ionicons name={icon as any} size={24} color={color} />
      </View>
      <Text style={styles.actionTitle}>{title}</Text>
    </TouchableOpacity>
  );
}

function createStyles(theme: any) {
  return StyleSheet.create({
    container: {
      paddingHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.xl,
    },
    sectionTitle: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    actionsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: theme.spacing.md,
    },
  });
}

function createActionStyles(theme: any, color: string) {
  return StyleSheet.create({
    actionButton: {
      flex: 1,
      minWidth: 80,
      alignItems: 'center',
      padding: theme.spacing.md,
      backgroundColor: theme.colors.surface,
      borderRadius: theme.borderRadius.md,
      shadowColor: theme.colors.text,
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    iconContainer: {
      width: 48,
      height: 48,
      borderRadius: 24,
      backgroundColor: `${color}15`,
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: theme.spacing.sm,
    },
    actionTitle: {
      fontSize: theme.typography.caption.fontSize,
      fontWeight: '600',
      color: theme.colors.text,
      textAlign: 'center',
    },
  });
}
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

interface QuickActionProps {
  title: string;
  icon: string;
  onPress: () => void;
  color?: string;
}

export function QuickActions(): JSX.Element {
  const { theme } = useTheme();
  const styles = createStyles(theme);

  const actions: QuickActionProps[] = [
    {
      title: 'Start Task',
      icon: 'play-outline',
      onPress: () => console.log('Start Task'),
      color: theme.colors.success,
    },
    {
      title: 'Stop Task',
      icon: 'stop-outline',
      onPress: () => console.log('Stop Task'),
      color: theme.colors.error,
    },
    {
      title: 'Restart',
      icon: 'refresh-outline',
      onPress: () => console.log('Restart'),
      color: theme.colors.warning,
    },
    {
      title: 'Settings',
      icon: 'settings-outline',
      onPress: () => console.log('Settings'),
      color: theme.colors.info,
    },
  ];

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Quick Actions</Text>
      <View style={styles.actionsGrid}>
        {actions.map((action, index) => (
          <QuickAction
            key={index}
            title={action.title}
            icon={action.icon}
            onPress={action.onPress}
            color={action.color}
          />
        ))}
      </View>
    </View>
  );
}

function QuickAction({ title, icon, onPress, color }: QuickActionProps): JSX.Element {
  const { theme } = useTheme();
  const styles = createActionStyles(theme, color);

  return (
    <TouchableOpacity
      style={styles.actionButton}
      onPress={onPress}
      accessible={true}
      accessibilityLabel={title}
      accessibilityRole="button"
    >
      <View style={styles.iconContainer}>
        <Ionicons name={icon as any} size={24} color={color} />
      </View>
      <Text style={styles.actionTitle}>{title}</Text>
    </TouchableOpacity>
  );
}

function createStyles(theme: any) {
  return StyleSheet.create({
    container: {
      paddingHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.xl,
    },
    sectionTitle: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    actionsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: theme.spacing.md,
    },
  });
}

function createActionStyles(theme: any, color: string) {
  return StyleSheet.create({
    actionButton: {
      flex: 1,
      minWidth: 80,
      alignItems: 'center',
      padding: theme.spacing.md,
      backgroundColor: theme.colors.surface,
      borderRadius: theme.borderRadius.md,
      shadowColor: theme.colors.text,
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    iconContainer: {
      width: 48,
      height: 48,
      borderRadius: 24,
      backgroundColor: `${color}15`,
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: theme.spacing.sm,
    },
    actionTitle: {
      fontSize: theme.typography.caption.fontSize,
      fontWeight: '600',
      color: theme.colors.text,
      textAlign: 'center',
    },
  });
}


