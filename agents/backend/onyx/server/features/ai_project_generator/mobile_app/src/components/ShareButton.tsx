import React from 'react';
import { TouchableOpacity, Text, StyleSheet, Share, Alert } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';
import { Project } from '../types';

interface ShareButtonProps {
  project: Project;
  variant?: 'icon' | 'button';
}

export const ShareButton: React.FC<ShareButtonProps> = ({
  project,
  variant = 'icon',
}) => {
  const { theme } = useTheme();

  const handleShare = async () => {
    try {
      hapticFeedback.selection();
      const result = await Share.share({
        message: `Proyecto: ${project.project_name}\n\n${project.description}\n\nEstado: ${project.status}\nAutor: ${project.author}\n\nID: ${project.project_id}`,
        title: project.project_name,
      });

      if (result.action === Share.sharedAction) {
        hapticFeedback.success();
      }
    } catch (error: any) {
      hapticFeedback.error();
      Alert.alert('Error', 'No se pudo compartir el proyecto');
    }
  };

  if (variant === 'button') {
    return (
      <TouchableOpacity
        style={[styles.button, { backgroundColor: theme.primary }]}
        onPress={handleShare}
        activeOpacity={0.7}
      >
        <Text style={styles.buttonIcon}>📤</Text>
        <Text style={[styles.buttonText, { color: theme.surface }]}>
          Compartir
        </Text>
      </TouchableOpacity>
    );
  }

  return (
    <TouchableOpacity
      style={styles.iconButton}
      onPress={handleShare}
      activeOpacity={0.7}
    >
      <Text style={[styles.icon, { fontSize: 24 }]}>📤</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
  },
  buttonIcon: {
    fontSize: 18,
  },
  buttonText: {
    ...typography.body,
    fontWeight: '600',
  },
  iconButton: {
    padding: spacing.sm,
  },
  icon: {
    textAlign: 'center',
  },
});

