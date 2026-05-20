import React, { useState } from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  Alert,
  Share,
  ActivityIndicator,
} from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { useActionHistory } from '../hooks/useActionHistory';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';
import { formatDateTime } from '../utils/date';

export const ExportHistoryButton: React.FC = () => {
  const { theme } = useTheme();
  const { history } = useActionHistory();
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    if (history.length === 0) {
      Alert.alert('Sin historial', 'No hay acciones para exportar');
      return;
    }

    setExporting(true);
    hapticFeedback.selection();

    try {
      const historyText = history
        .map((item) => {
          const date = formatDateTime(new Date(item.timestamp));
          const status = item.success ? '✓' : '✗';
          return `${date} ${status} ${item.type} - ${item.projectName || item.projectId || 'N/A'}`;
        })
        .join('\n');

      const exportText = `Historial de Acciones\n\nTotal: ${history.length} acciones\n\n${historyText}`;

      await Share.share({
        message: exportText,
        title: 'Historial de Acciones',
      });

      hapticFeedback.success();
    } catch (error: any) {
      hapticFeedback.error();
      Alert.alert('Error', 'No se pudo exportar el historial');
    } finally {
      setExporting(false);
    }
  };

  return (
    <TouchableOpacity
      style={[styles.button, { backgroundColor: theme.primary }]}
      onPress={handleExport}
      disabled={exporting || history.length === 0}
      activeOpacity={0.7}
    >
      {exporting ? (
        <ActivityIndicator color={theme.surface} />
      ) : (
        <>
          <Text style={styles.icon}>📥</Text>
          <Text style={[styles.text, { color: theme.surface }]}>
            Exportar Historial ({history.length})
          </Text>
        </>
      )}
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
  icon: {
    fontSize: 18,
  },
  text: {
    ...typography.body,
    fontWeight: '600',
  },
});

