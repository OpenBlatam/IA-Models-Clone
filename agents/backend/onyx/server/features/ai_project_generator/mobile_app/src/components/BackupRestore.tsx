import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Share,
  ActivityIndicator,
} from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { storage, STORAGE_KEYS } from '../utils/storage';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';
import { useToastHelpers } from '../hooks/useToast';

export const BackupRestore: React.FC = () => {
  const { theme } = useTheme();
  const toast = useToastHelpers();
  const [backingUp, setBackingUp] = useState(false);
  const [restoring, setRestoring] = useState(false);

  const handleBackup = async () => {
    setBackingUp(true);
    hapticFeedback.selection();

    try {
      const backupData: Record<string, any> = {};

      const keys = [
        STORAGE_KEYS.USER_PREFERENCES,
        STORAGE_KEYS.FAVORITES,
        STORAGE_KEYS.ACTION_HISTORY,
        STORAGE_KEYS.THEME_MODE,
      ];

      for (const key of keys) {
        const value = await storage.get(key);
        if (value) {
          backupData[key] = value;
        }
      }

      backupData.timestamp = Date.now();
      backupData.version = '1.0.0';

      await storage.set(STORAGE_KEYS.BACKUP_DATA, backupData);

      const backupText = JSON.stringify(backupData, null, 2);

      await Share.share({
        message: backupText,
        title: 'Backup de Datos',
      });

      toast.showSuccess('Backup creado exitosamente');
      hapticFeedback.success();
    } catch (error: any) {
      hapticFeedback.error();
      toast.showError('Error al crear backup');
      Alert.alert('Error', 'No se pudo crear el backup');
    } finally {
      setBackingUp(false);
    }
  };

  const handleRestore = async () => {
    Alert.alert(
      'Restaurar Backup',
      'Esta acción sobrescribirá tus datos actuales. ¿Continuar?',
      [
        { text: 'Cancelar', style: 'cancel' },
        {
          text: 'Restaurar',
          style: 'destructive',
          onPress: async () => {
            setRestoring(true);
            hapticFeedback.selection();

            try {
              const backupData = await storage.get<Record<string, any>>(
                STORAGE_KEYS.BACKUP_DATA
              );

              if (!backupData) {
                Alert.alert('Error', 'No se encontró un backup para restaurar');
                return;
              }

              for (const [key, value] of Object.entries(backupData)) {
                if (key !== 'timestamp' && key !== 'version') {
                  await storage.set(key, value);
                }
              }

              toast.showSuccess('Backup restaurado exitosamente');
              hapticFeedback.success();
            } catch (error: any) {
              hapticFeedback.error();
              toast.showError('Error al restaurar backup');
              Alert.alert('Error', 'No se pudo restaurar el backup');
            } finally {
              setRestoring(false);
            }
          },
        },
      ]
    );
  };

  return (
    <View style={styles.container}>
      <Text style={[styles.title, { color: theme.text }]}>Backup y Restauración</Text>
      <Text style={[styles.description, { color: theme.textSecondary }]}>
        Crea un respaldo de tus datos o restaura un backup anterior
      </Text>

      <View style={styles.buttons}>
        <TouchableOpacity
          style={[styles.button, { backgroundColor: theme.primary }]}
          onPress={handleBackup}
          disabled={backingUp || restoring}
          activeOpacity={0.7}
        >
          {backingUp ? (
            <ActivityIndicator color={theme.surface} />
          ) : (
            <>
              <Text style={styles.icon}>💾</Text>
              <Text style={[styles.buttonText, { color: theme.surface }]}>Crear Backup</Text>
            </>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.restoreButton, { borderColor: theme.primary }]}
          onPress={handleRestore}
          disabled={backingUp || restoring}
          activeOpacity={0.7}
        >
          {restoring ? (
            <ActivityIndicator color={theme.primary} />
          ) : (
            <>
              <Text style={styles.icon}>📥</Text>
              <Text style={[styles.buttonText, styles.restoreButtonText, { color: theme.primary }]}>
                Restaurar Backup
              </Text>
            </>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: spacing.xl,
  },
  title: {
    ...typography.h3,
    marginBottom: spacing.sm,
  },
  description: {
    ...typography.bodySmall,
    marginBottom: spacing.xl,
    lineHeight: 20,
  },
  buttons: {
    gap: spacing.md,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
  },
  restoreButton: {
    backgroundColor: 'transparent',
    borderWidth: 2,
  },
  icon: {
    fontSize: 18,
  },
  buttonText: {
    ...typography.body,
    fontWeight: '600',
  },
  restoreButtonText: {
    fontWeight: '600',
  },
});

