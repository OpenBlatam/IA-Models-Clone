import React, { memo } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Project } from '../types';
import { StatusBadge } from './StatusBadge';
import { SwipeableCard } from './SwipeableCard';
import { FavoriteButton } from './FavoriteButton';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { formatDate } from '../utils/date';

interface ProjectCardProps {
  project: Project;
  onPress?: () => void;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  enableSwipe?: boolean;
}

const ProjectCardContent: React.FC<{
  project: Project;
  onPress?: () => void;
}> = memo(({ project, onPress }) => {
  const { theme } = useTheme();
  
  return (
    <TouchableOpacity
      style={[styles.card, { backgroundColor: theme.surface, shadowColor: theme.shadow }]}
      onPress={onPress}
      activeOpacity={0.7}
    >
      <View style={styles.header}>
        <View style={styles.titleContainer}>
          <Text style={[styles.title, { color: theme.text }]} numberOfLines={1}>
            {project.project_name}
          </Text>
          <FavoriteButton projectId={project.project_id} size={20} />
        </View>
        <StatusBadge status={project.status} />
      </View>
      <Text style={[styles.description, { color: theme.textSecondary }]} numberOfLines={2}>
        {project.description}
      </Text>
      <View style={styles.footer}>
        <Text style={[styles.author, { color: theme.textTertiary }]}>Por: {project.author}</Text>
        <Text style={[styles.date, { color: theme.textTertiary }]}>{formatDate(project.created_at)}</Text>
      </View>
    </TouchableOpacity>
  );
});

ProjectCardContent.displayName = 'ProjectCardContent';

export const ProjectCard: React.FC<ProjectCardProps> = memo(
  ({ project, onPress, onSwipeLeft, onSwipeRight, enableSwipe = false }) => {
    if (enableSwipe && (onSwipeLeft || onSwipeRight)) {
      return (
        <SwipeableCard
          onSwipeLeft={onSwipeLeft}
          onSwipeRight={onSwipeRight}
          leftAction={
            onSwipeLeft
              ? {
                  label: 'Eliminar',
                  color: colors.error,
                  icon: '🗑️',
                }
              : undefined
          }
          rightAction={
            onSwipeRight
              ? {
                  label: 'Ver',
                  color: colors.primary,
                  icon: '👁️',
                }
              : undefined
          }
        >
          <ProjectCardContent project={project} onPress={onPress} />
        </SwipeableCard>
      );
    }

    return <ProjectCardContent project={project} onPress={onPress} />;
  }
);

ProjectCard.displayName = 'ProjectCard';

const styles = StyleSheet.create({
  card: {
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  titleContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: spacing.sm,
    gap: spacing.xs,
  },
  title: {
    ...typography.h3,
    flex: 1,
  },
  description: {
    ...typography.bodySmall,
    marginBottom: spacing.md,
    lineHeight: 20,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  author: {
    ...typography.caption,
  },
  date: {
    ...typography.caption,
  },
});
