import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  TextInput,
  Alert,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { storage, STORAGE_KEYS } from '../utils/storage';
import { useToastHelpers } from '../hooks/useToast';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { hapticFeedback } from '../utils/haptics';

type ThemeMode = 'light' | 'dark' | 'auto';

interface UserPreferences {
  apiUrl?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
  notifications?: boolean;
}

export const SettingsScreen: React.FC = () => {
  const navigation = useNavigation();
  const { theme, isDark, themeMode, setThemeMode, toggleTheme } = useTheme();
  const toast = useToastHelpers();
  const [preferences, setPreferences] = useState<UserPreferences>({
    autoRefresh: true,
    refreshInterval: 30,
    notifications: true,
  });
  const [apiUrl, setApiUrl] = useState('');
  const [showClearCacheConfirm, setShowClearCacheConfirm] = useState(false);

  React.useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    try {
      const saved = await storage.get<UserPreferences>(
        STORAGE_KEYS.USER_PREFERENCES
      );
      if (saved) {
        setPreferences(saved);
        setApiUrl(saved.apiUrl || '');
      }
    } catch (error) {
      console.error('Error loading preferences:', error);
    }
  };

  const savePreferences = async () => {
    try {
      const newPreferences: UserPreferences = {
        ...preferences,
        apiUrl: apiUrl.trim() || undefined,
      };
      await storage.set(STORAGE_KEYS.USER_PREFERENCES, newPreferences);
      toast.showSuccess('Preferencias guardadas');
    } catch (error) {
      toast.showError('Error al guardar preferencias');
    }
  };

  const handleClearCache = async () => {
    try {
      await storage.remove(STORAGE_KEYS.CACHED_PROJECTS);
      await storage.remove(STORAGE_KEYS.CACHED_STATS);
      await storage.remove(STORAGE_KEYS.LAST_SYNC);
      toast.showSuccess('Caché limpiado correctamente');
      setShowClearCacheConfirm(false);
    } catch (error) {
      toast.showError('Error al limpiar caché');
    }
  };

  const handleThemeToggle = () => {
    hapticFeedback.selection();
    toggleTheme();
  };

  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.background }]}>
      <View style={[styles.header, { backgroundColor: theme.surface, borderBottomColor: theme.border }]}>
        <Text style={[styles.title, { color: theme.text }]}>Configuración</Text>
      </View>

      <View style={[styles.section, { backgroundColor: theme.surface, borderColor: theme.border }]}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Apariencia</Text>
        <View style={styles.switchGroup}>
          <View style={styles.switchLabelContainer}>
            <Text style={[styles.switchLabel, { color: theme.text }]}>Modo Oscuro</Text>
            <Text style={[styles.switchDescription, { color: theme.textSecondary }]}>
              {themeMode === 'auto' ? 'Sigue el sistema' : isDark ? 'Activado' : 'Desactivado'}
            </Text>
          </View>
          <Switch
            value={isDark}
            onValueChange={handleThemeToggle}
            trackColor={{ false: theme.border, true: theme.primary }}
            thumbColor={theme.surface}
          />
        </View>
        <TouchableOpacity
          style={[styles.themeModeButton, { backgroundColor: theme.surfaceVariant }]}
          onPress={() => {
            const modes: ThemeMode[] = ['light', 'dark', 'auto'];
            const currentIndex = modes.indexOf(themeMode);
            const nextMode = modes[(currentIndex + 1) % modes.length];
            setThemeMode(nextMode);
            hapticFeedback.selection();
          }}
        >
          <Text style={[styles.themeModeText, { color: theme.text }]}>
            Modo: {themeMode === 'auto' ? 'Automático' : themeMode === 'dark' ? 'Oscuro' : 'Claro'}
          </Text>
        </TouchableOpacity>
      </View>

      <View style={[styles.section, { backgroundColor: theme.surface, borderColor: theme.border }]}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>API</Text>
        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>URL del Servidor</Text>
          <TextInput
            style={[styles.input, { backgroundColor: theme.surfaceVariant, borderColor: theme.border, color: theme.text }]}
            placeholder="http://localhost:8020"
            placeholderTextColor={theme.textTertiary}
            value={apiUrl}
            onChangeText={setApiUrl}
            autoCapitalize="none"
            keyboardType="url"
          />
          <Text style={[styles.hint, { color: theme.textSecondary }]}>
            Cambia la URL del servidor API. Requiere reiniciar la app.
          </Text>
        </View>
      </View>

      <View style={[styles.section, { backgroundColor: theme.surface, borderColor: theme.border }]}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Actualización</Text>
        <View style={styles.switchGroup}>
          <View style={styles.switchLabelContainer}>
            <Text style={[styles.switchLabel, { color: theme.text }]}>Actualización Automática</Text>
            <Text style={[styles.switchDescription, { color: theme.textSecondary }]}>
              Actualiza los datos automáticamente
            </Text>
          </View>
          <Switch
            value={preferences.autoRefresh}
            onValueChange={(value) =>
              setPreferences({ ...preferences, autoRefresh: value })
            }
            trackColor={{ false: theme.border, true: theme.primary }}
            thumbColor={theme.surface}
          />
        </View>
        {preferences.autoRefresh && (
          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: theme.text }]}>Intervalo (segundos)</Text>
            <TextInput
              style={[styles.input, { backgroundColor: theme.surfaceVariant, borderColor: theme.border, color: theme.text }]}
              placeholder="30"
              placeholderTextColor={theme.textTertiary}
              value={preferences.refreshInterval?.toString() || '30'}
              onChangeText={(text) => {
                const num = parseInt(text, 10);
                if (!isNaN(num) && num > 0) {
                  setPreferences({ ...preferences, refreshInterval: num });
                }
              }}
              keyboardType="numeric"
            />
          </View>
        )}
      </View>

      <View style={[styles.section, { backgroundColor: theme.surface, borderColor: theme.border }]}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Notificaciones</Text>
        <View style={styles.switchGroup}>
          <View style={styles.switchLabelContainer}>
            <Text style={[styles.switchLabel, { color: theme.text }]}>Notificaciones</Text>
            <Text style={[styles.switchDescription, { color: theme.textSecondary }]}>
              Recibe notificaciones cuando los proyectos se completen
            </Text>
          </View>
          <Switch
            value={preferences.notifications}
            onValueChange={(value) =>
              setPreferences({ ...preferences, notifications: value })
            }
            trackColor={{ false: theme.border, true: theme.primary }}
            thumbColor={theme.surface}
          />
        </View>
      </View>

      <View style={[styles.section, { backgroundColor: theme.surface, borderColor: theme.border }]}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Almacenamiento</Text>
        <TouchableOpacity
          style={[styles.actionButton, { backgroundColor: theme.surfaceVariant, borderColor: theme.border }]}
          onPress={() => setShowClearCacheConfirm(true)}
        >
          <Text style={[styles.actionButtonText, { color: theme.text }]}>🗑️ Limpiar Caché</Text>
        </TouchableOpacity>
        <Text style={[styles.hint, { color: theme.textSecondary }]}>
          Esto eliminará todos los datos en caché. Los datos se volverán a
          descargar cuando sea necesario.
        </Text>
      </View>

      <View style={[styles.section, { backgroundColor: theme.surface, borderColor: theme.border }]}>
        <Text style={[styles.sectionTitle, { color: theme.text }]}>Información</Text>
        <View style={styles.infoRow}>
          <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>Versión:</Text>
          <Text style={[styles.infoValue, { color: theme.text }]}>1.0.0</Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={[styles.infoLabel, { color: theme.textSecondary }]}>Plataforma:</Text>
          <Text style={[styles.infoValue, { color: theme.text }]}>React Native + Expo</Text>
        </View>
      </View>

      <TouchableOpacity style={[styles.saveButton, { backgroundColor: theme.primary }]} onPress={savePreferences}>
        <Text style={[styles.saveButtonText, { color: theme.surface }]}>💾 Guardar Configuración</Text>
      </TouchableOpacity>

      <ConfirmDialog
        visible={showClearCacheConfirm}
        title="Limpiar Caché"
        message="¿Estás seguro de que quieres limpiar todo el caché? Esto eliminará todos los datos almacenados localmente."
        confirmText="Limpiar"
        cancelText="Cancelar"
        type="warning"
        onConfirm={handleClearCache}
        onCancel={() => setShowClearCacheConfirm(false)}
      />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    padding: spacing.xl,
    borderBottomWidth: 1,
  },
  title: {
    ...typography.h2,
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
  themeModeButton: {
    borderWidth: 1,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    alignItems: 'center',
    marginTop: spacing.sm,
  },
  themeModeText: {
    ...typography.body,
    fontWeight: '600',
  },
  inputGroup: {
    marginBottom: spacing.lg,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '600',
    color: colors.text,
    marginBottom: spacing.sm,
  },
  input: {
    backgroundColor: colors.surfaceVariant,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    ...typography.body,
    color: colors.text,
  },
  hint: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: spacing.xs,
  },
  switchGroup: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
    paddingVertical: spacing.sm,
  },
  switchLabelContainer: {
    flex: 1,
    marginRight: spacing.md,
  },
  switchLabel: {
    ...typography.body,
    fontWeight: '600',
    color: colors.text,
    marginBottom: spacing.xs,
  },
  switchDescription: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  actionButton: {
    backgroundColor: colors.surfaceVariant,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  actionButtonText: {
    ...typography.body,
    fontWeight: '600',
    color: colors.text,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
  },
  infoLabel: {
    ...typography.body,
    color: colors.textSecondary,
  },
  infoValue: {
    ...typography.body,
    fontWeight: '600',
    color: colors.text,
  },
  saveButton: {
    backgroundColor: colors.primary,
    margin: spacing.xl,
    padding: spacing.lg,
    borderRadius: borderRadius.lg,
    alignItems: 'center',
  },
  saveButtonText: {
    ...typography.body,
    fontWeight: '600',
    color: colors.surface,
  },
});

