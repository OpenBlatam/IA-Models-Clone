import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { useRoute, useNavigation } from '@react-navigation/native';
import { useProjectQuery, useDeleteProjectMutation } from '../hooks/useProjectsQuery';
import { useToastHelpers } from '../hooks/useToast';
import { useTheme } from '../contexts/ThemeContext';
import { useAnalytics } from '../hooks/useAnalytics';
import { StatusBadge } from '../components/StatusBadge';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { ShareButton } from '../components/ShareButton';
import { CopyButton } from '../components/CopyButton';
import { Divider } from '../components/Divider';
import { Badge } from '../components/Badge';
import { spacing, borderRadius, typography } from '../theme/colors';
import { formatDateTime } from '../utils/date';
import { apiService } from '../services/api';

export const ProjectDetailScreen: React.FC = () => {
  const route = useRoute();
  const navigation = useNavigation();
  const { theme } = useTheme();
  const analytics = useAnalytics();
  const { projectId } = route.params as { projectId: string };
  const [exporting, setExporting] = useState<'zip' | 'tar' | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const {
    data: project,
    isLoading,
    error,
    refetch,
  } = useProjectQuery(projectId);

  const deleteMutation = useDeleteProjectMutation();
  const toast = useToastHelpers();

  React.useEffect(() => {
    if (project) {
      analytics.trackScreenView('ProjectDetailScreen', { projectId: project.project_id });
    }
  }, [project]);

  const handleDelete = () => {
    setShowDeleteConfirm(true);
  };

  const confirmDelete = async () => {
    setShowDeleteConfirm(false);
    try {
      await deleteMutation.mutateAsync(projectId);
      toast.showSuccess('Proyecto eliminado correctamente');
      navigation.goBack();
    } catch (err: any) {
      toast.showError(err.detail || 'Error al eliminar el proyecto');
    }
  };

  const handleExport = async (format: 'zip' | 'tar') => {
    setExporting(format);
    try {
      if (format === 'zip') {
        await apiService.exportZip(projectId);
      } else {
        await apiService.exportTar(projectId);
      }
      toast.showSuccess(`Proyecto exportado como ${format.toUpperCase()}`);
    } catch (err: any) {
      toast.showError(err.detail || 'Error al exportar el proyecto');
    } finally {
      setExporting(null);
    }
  };

  const handleValidate = async () => {
    try {
      const result = await apiService.validateProject(projectId);
      if (result.valid) {
        toast.showSuccess(
          `Proyecto válido${result.score ? ` (Score: ${result.score})` : ''}`
        );
        if (result.warnings.length > 0) {
          setTimeout(() => {
            toast.showWarning(
              `Advertencias: ${result.warnings.length} encontradas`
            );
          }, 500);
        }
      } else {
        toast.showError(`Errores encontrados: ${result.errors.length}`);
      }
    } catch (err: any) {
      toast.showError(err.detail || 'Error al validar el proyecto');
    }
  };

  if (isLoading) {
    return <LoadingSpinner message="Cargando proyecto..." />;
  }

  if (error) {
    return <ErrorMessage error={error} onRetry={() => refetch()} />;
  }

  if (!project) {
    return null;
  }

  return (
    <>
      <ScrollView style={[styles.container, { backgroundColor: theme.background }]}>
        <View style={[styles.header, { backgroundColor: theme.surface, borderBottomColor: theme.border }]}>
          <View style={styles.headerContent}>
            <Text style={[styles.title, { color: theme.text }]} numberOfLines={2}>
              {project.project_name}
            </Text>
            <StatusBadge status={project.status} />
          </View>
          <ShareButton
            project={project}
            onShare={() => analytics.trackUserAction('share_project', { projectId: project.project_id })}
          />
        </View>

        <View style={[styles.section, { backgroundColor: theme.surface, borderColor: theme.border }]}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Descripción</Text>
          <Text style={[styles.description, { color: theme.textSecondary }]}>{project.description}</Text>
        </View>

        <Divider />

        <View style={[styles.section, { backgroundColor: theme.surface, borderColor: theme.border }]}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Información</Text>
          <View style={styles.infoRow}>
            <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>Autor:</Text>
            <Text style={[styles.infoValue, { color: theme.text }]}>{project.author}</Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>ID:</Text>
            <View style={styles.infoValueContainer}>
              <Text style={[styles.infoValue, styles.infoValueSmall, { color: theme.text }]}>
                {project.project_id}
              </Text>
              <CopyButton text={project.project_id} />
            </View>
          </View>
          <View style={styles.infoRow}>
            <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>Estado:</Text>
            <StatusBadge status={project.status} />
          </View>
          <View style={styles.infoRow}>
            <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>Creado:</Text>
            <Text style={[styles.infoValue, { color: theme.text }]}>
              {formatDateTime(project.created_at)}
            </Text>
          </View>
          {project.updated_at && (
            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>Actualizado:</Text>
              <Text style={[styles.infoValue, { color: theme.text }]}>
                {formatDateTime(project.updated_at)}
              </Text>
            </View>
          )}
          {project.project_dir && (
            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>Directorio:</Text>
              <View style={styles.infoValueContainer}>
                <Text style={[styles.infoValue, styles.infoValueSmall, { color: theme.text }]}>
                  {project.project_dir}
                </Text>
                <CopyButton text={project.project_dir} />
              </View>
            </View>
          )}
        </View>

        {project.metadata && Object.keys(project.metadata).length > 0 && (
          <>
            <Divider />
            <View style={[styles.section, { backgroundColor: theme.surface, borderColor: theme.border }]}>
              <Text style={[styles.sectionTitle, { color: theme.text }]}>Metadata</Text>
              {project.metadata.tags && Array.isArray(project.metadata.tags) && project.metadata.tags.length > 0 && (
                <View style={styles.tagsContainer}>
                  {project.metadata.tags.map((tag: string, index: number) => (
                    <Badge key={index} label={tag} variant="info" size="small" />
                  ))}
                </View>
              )}
              <View style={[styles.metadataContainer, { backgroundColor: theme.surfaceVariant }]}>
                <Text style={[styles.metadata, { color: theme.textSecondary }]}>
                  {JSON.stringify(project.metadata, null, 2)}
                </Text>
                <CopyButton
                  text={JSON.stringify(project.metadata, null, 2)}
                  label="Copiar JSON"
                  showLabel={true}
                />
              </View>
            </View>
          </>
        )}

        <View style={styles.actions}>
          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: theme.success }]}
            onPress={() => {
              analytics.trackUserAction('validate_project', { projectId: project.project_id });
              handleValidate();
            }}
          >
            <Text style={[styles.actionButtonText, { color: theme.surface }]}>✓ Validar Proyecto</Text>
          </TouchableOpacity>

          <View style={styles.exportButtons}>
            <TouchableOpacity
              style={[
                styles.actionButton,
                styles.exportButton,
                { backgroundColor: theme.primary },
                exporting === 'zip' && styles.actionButtonDisabled,
              ]}
              onPress={() => {
                analytics.trackUserAction('export_project', { projectId: project.project_id, format: 'zip' });
                handleExport('zip');
              }}
              disabled={!!exporting}
            >
              {exporting === 'zip' ? (
                <ActivityIndicator color={theme.surface} />
              ) : (
                <Text style={[styles.actionButtonText, { color: theme.surface }]}>📦 Exportar ZIP</Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.actionButton,
                styles.exportButton,
                { backgroundColor: theme.primary },
                exporting === 'tar' && styles.actionButtonDisabled,
              ]}
              onPress={() => {
                analytics.trackUserAction('export_project', { projectId: project.project_id, format: 'tar' });
                handleExport('tar');
              }}
              disabled={!!exporting}
            >
              {exporting === 'tar' ? (
                <ActivityIndicator color={theme.surface} />
              ) : (
                <Text style={[styles.actionButtonText, { color: theme.surface }]}>📦 Exportar TAR</Text>
              )}
            </TouchableOpacity>
          </View>

          <TouchableOpacity
            style={[
              styles.actionButton,
              styles.deleteButton,
              { backgroundColor: theme.surface, borderColor: theme.error },
            ]}
            onPress={() => {
              analytics.trackUserAction('delete_project', { projectId: project.project_id });
              handleDelete();
            }}
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? (
              <ActivityIndicator color={theme.error} />
            ) : (
              <Text style={[styles.actionButtonText, styles.deleteButtonText, { color: theme.error }]}>
                🗑️ Eliminar
              </Text>
            )}
          </TouchableOpacity>
        </View>
      </ScrollView>

      <ConfirmDialog
        visible={showDeleteConfirm}
        title="Eliminar Proyecto"
        message="¿Estás seguro de que quieres eliminar este proyecto? Esta acción no se puede deshacer."
        confirmText="Eliminar"
        cancelText="Cancelar"
        type="danger"
        onConfirm={confirmDelete}
        onCancel={() => setShowDeleteConfirm(false)}
      />
    </>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    padding: spacing.xl,
    borderBottomWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerContent: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  title: {
    ...typography.h2,
    flex: 1,
  },
  section: {
    padding: spacing.xl,
    marginTop: spacing.md,
    borderTopWidth: 1,
    borderBottomWidth: 1,
  },
  sectionTitle: {
    ...typography.h3,
    marginBottom: spacing.md,
  },
  description: {
    ...typography.body,
    lineHeight: 24,
  },
  infoRow: {
    flexDirection: 'row',
    marginBottom: spacing.md,
    alignItems: 'center',
  },
  infoLabel: {
    ...typography.bodySmall,
    fontWeight: '600',
    width: 100,
  },
  infoValue: {
    ...typography.bodySmall,
    flex: 1,
  },
  infoValueSmall: {
    fontSize: 11,
  },
  infoValueContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  metadataContainer: {
    padding: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.md,
  },
  metadata: {
    ...typography.caption,
    fontFamily: 'monospace',
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  actions: {
    padding: spacing.xl,
    gap: spacing.md,
  },
  exportButtons: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  actionButton: {
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 50,
  },
  exportButton: {
    flex: 1,
  },
  deleteButton: {
    borderWidth: 2,
  },
  actionButtonDisabled: {
    opacity: 0.6,
  },
  actionButtonText: {
    ...typography.body,
    fontWeight: '600',
  },
  deleteButtonText: {
  },
});
