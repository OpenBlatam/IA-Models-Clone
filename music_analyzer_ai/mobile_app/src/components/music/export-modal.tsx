import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  ScrollView,
  Alert,
  Share,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useExportAnalysis } from '../../hooks/use-music-api';
import { LoadingSpinner } from '../common/loading-spinner';
import { ErrorMessage } from '../common/error-message';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useTranslation } from '../../hooks/use-translation';
import { useToast } from '../../contexts/toast-context';

interface ExportModalProps {
  visible: boolean;
  trackId: string;
  trackName: string;
  onClose: () => void;
}

type ExportFormat = 'json' | 'text' | 'markdown';

export function ExportModal({
  visible,
  trackId,
  trackName,
  onClose,
}: ExportModalProps) {
  const { t } = useTranslation();
  const { showToast } = useToast();
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('json');
  const [includeCoaching, setIncludeCoaching] = useState(true);
  const exportMutation = useExportAnalysis();

  const handleExport = useCallback(() => {
    exportMutation.mutate(
      {
        trackId,
        format: selectedFormat,
        includeCoaching,
      },
      {
        onSuccess: async (data) => {
          try {
            await Share.share({
              message: data.content,
              title: `Analysis: ${trackName}`,
            });
            showToast('Analysis exported and shared successfully!', 'success');
            onClose();
          } catch (error) {
            showToast('Failed to share analysis', 'error');
          }
        },
        onError: (error) => {
          showToast(error.message || 'Failed to export analysis', 'error');
        },
      }
    );
  }, [trackId, selectedFormat, includeCoaching, exportMutation, trackName, onClose]);

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent={true}
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <SafeAreaView style={styles.container} edges={['bottom']}>
          <View style={styles.header}>
            <Text style={styles.title}>Export Analysis</Text>
            <TouchableOpacity onPress={onClose} style={styles.closeButton}>
              <Text style={styles.closeButtonText}>✕</Text>
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.content}>
            <Text style={styles.label}>Format</Text>
            <View style={styles.formatContainer}>
              {(['json', 'text', 'markdown'] as ExportFormat[]).map((format) => (
                <TouchableOpacity
                  key={format}
                  style={[
                    styles.formatButton,
                    selectedFormat === format && styles.formatButtonActive,
                  ]}
                  onPress={() => setSelectedFormat(format)}
                  accessibilityRole="button"
                >
                  <Text
                    style={[
                      styles.formatButtonText,
                      selectedFormat === format && styles.formatButtonTextActive,
                    ]}
                  >
                    {format.toUpperCase()}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <View style={styles.optionContainer}>
              <TouchableOpacity
                style={styles.checkboxContainer}
                onPress={() => setIncludeCoaching(!includeCoaching)}
                accessibilityRole="checkbox"
                accessibilityState={{ checked: includeCoaching }}
              >
                <View
                  style={[
                    styles.checkbox,
                    includeCoaching && styles.checkboxChecked,
                  ]}
                >
                  {includeCoaching && (
                    <Text style={styles.checkmark}>✓</Text>
                  )}
                </View>
                <Text style={styles.checkboxLabel}>Include Coaching</Text>
              </TouchableOpacity>
            </View>

            {exportMutation.isError && (
              <ErrorMessage
                message={
                  exportMutation.error?.message || 'Failed to export analysis'
                }
                onRetry={() => exportMutation.reset()}
              />
            )}
          </ScrollView>

          <View style={styles.footer}>
            <TouchableOpacity
              style={styles.cancelButton}
              onPress={onClose}
              accessibilityRole="button"
            >
              <Text style={styles.cancelButtonText}>{t('common.cancel')}</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.exportButton,
                exportMutation.isPending && styles.exportButtonDisabled,
              ]}
              onPress={handleExport}
              disabled={exportMutation.isPending}
              accessibilityRole="button"
            >
              {exportMutation.isPending ? (
                <LoadingSpinner size="small" color={COLORS.text} />
              ) : (
                <Text style={styles.exportButtonText}>
                  {t('common.save')} & Share
                </Text>
              )}
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  container: {
    backgroundColor: COLORS.surface,
    borderTopLeftRadius: BORDER_RADIUS.xl,
    borderTopRightRadius: BORDER_RADIUS.xl,
    maxHeight: '80%',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  title: {
    ...TYPOGRAPHY.h2,
    color: COLORS.text,
  },
  closeButton: {
    padding: SPACING.xs,
  },
  closeButtonText: {
    ...TYPOGRAPHY.h2,
    color: COLORS.textSecondary,
  },
  content: {
    padding: SPACING.md,
  },
  label: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    marginBottom: SPACING.sm,
    fontWeight: '600',
  },
  formatContainer: {
    flexDirection: 'row',
    gap: SPACING.sm,
    marginBottom: SPACING.lg,
  },
  formatButton: {
    flex: 1,
    padding: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    backgroundColor: COLORS.surfaceLight,
    alignItems: 'center',
  },
  formatButtonActive: {
    backgroundColor: COLORS.primary,
  },
  formatButtonText: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
    fontWeight: '600',
  },
  formatButtonTextActive: {
    color: COLORS.text,
  },
  optionContainer: {
    marginBottom: SPACING.lg,
  },
  checkboxContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: BORDER_RADIUS.sm,
    borderWidth: 2,
    borderColor: COLORS.textSecondary,
    marginRight: SPACING.sm,
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkboxChecked: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  checkmark: {
    color: COLORS.text,
    fontSize: 16,
    fontWeight: '600',
  },
  checkboxLabel: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
  },
  footer: {
    flexDirection: 'row',
    padding: SPACING.md,
    borderTopWidth: 1,
    borderTopColor: COLORS.surfaceLight,
    gap: SPACING.sm,
  },
  cancelButton: {
    flex: 1,
    padding: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    backgroundColor: COLORS.surfaceLight,
    alignItems: 'center',
  },
  cancelButtonText: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    fontWeight: '600',
  },
  exportButton: {
    flex: 2,
    padding: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
  },
  exportButtonDisabled: {
    opacity: 0.6,
  },
  exportButtonText: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    fontWeight: '600',
  },
});

