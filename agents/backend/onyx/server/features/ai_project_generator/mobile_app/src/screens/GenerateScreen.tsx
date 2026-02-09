import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Switch,
  Alert,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useGenerateProjectMutation } from '../hooks/useProjectsQuery';
import { useToastHelpers } from '../hooks/useToast';
import { useTheme } from '../contexts/ThemeContext';
import { useAnalytics } from '../hooks/useAnalytics';
import { ProjectRequest } from '../types';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { spacing, borderRadius, typography } from '../theme/colors';

export const GenerateScreen: React.FC = () => {
  const navigation = useNavigation();
  const { theme } = useTheme();
  const generateMutation = useGenerateProjectMutation();
  const toast = useToastHelpers();
  const analytics = useAnalytics();

  React.useEffect(() => {
    analytics.trackScreenView('GenerateScreen');
  }, []);

  const [formData, setFormData] = useState<ProjectRequest>({
    description: '',
    project_name: '',
    author: 'Blatam Academy',
    version: '1.0.0',
    priority: 0,
    backend_framework: 'fastapi',
    frontend_framework: 'react',
    generate_tests: true,
    include_docker: true,
    include_docs: true,
    tags: [],
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const validateField = (field: string, value: any): string | null => {
    switch (field) {
      case 'description':
        if (!value || value.trim().length < 10) {
          return 'La descripción debe tener al menos 10 caracteres';
        }
        if (value.length > 2000) {
          return 'La descripción no puede exceder 2000 caracteres';
        }
        return null;
      case 'project_name':
        if (value && value.length < 3) {
          return 'El nombre debe tener al menos 3 caracteres';
        }
        if (value && value.length > 50) {
          return 'El nombre no puede exceder 50 caracteres';
        }
        if (value && !/^[a-z0-9_]+$/.test(value)) {
          return 'El nombre solo puede contener letras minúsculas, números y guiones bajos';
        }
        return null;
      case 'version':
        if (value && !/^\d+\.\d+\.\d+$/.test(value)) {
          return 'La versión debe seguir el formato semántico (ej: 1.0.0)';
        }
        return null;
      default:
        return null;
    }
  };

  const handleFieldChange = (field: string, value: any) => {
    setFormData({ ...formData, [field]: value });
    const error = validateField(field, value);
    if (error) {
      setErrors({ ...errors, [field]: error });
    } else {
      const newErrors = { ...errors };
      delete newErrors[field];
      setErrors(newErrors);
    }
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    const descriptionError = validateField('description', formData.description);
    if (descriptionError) newErrors.description = descriptionError;

    if (formData.project_name) {
      const nameError = validateField('project_name', formData.project_name);
      if (nameError) newErrors.project_name = nameError;
    }

    if (formData.version) {
      const versionError = validateField('version', formData.version);
      if (versionError) newErrors.version = versionError;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleGenerate = async () => {
    if (!validateForm()) {
      toast.showError('Por favor corrige los errores en el formulario');
      return;
    }

    try {
      const response = await generateMutation.mutateAsync({
        request: formData,
        asyncGeneration: true,
      });

      toast.showSuccess(`Proyecto creado: ${response.project_id}`);

      Alert.alert(
        'Éxito',
        `Proyecto creado: ${response.project_id}`,
        [
          {
            text: 'Ver Proyecto',
            onPress: () => {
              navigation.navigate('ProjectDetail' as never, {
                projectId: response.project_id,
              } as never);
            },
          },
          {
            text: 'Crear Otro',
            onPress: () => {
              setFormData({
                ...formData,
                description: '',
                project_name: '',
              });
              setErrors({});
            },
          },
          { text: 'OK' },
        ]
      );
    } catch (error: any) {
      toast.showError(error.detail || 'Error al generar el proyecto');
    }
  };

  if (generateMutation.isPending) {
    return <LoadingSpinner message="Generando proyecto..." />;
  }

  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.background }]}>
      <View style={[styles.header, { backgroundColor: theme.surface, borderBottomColor: theme.border }]}>
        <Text style={[styles.title, { color: theme.text }]}>Generar Proyecto</Text>
        <Text style={[styles.subtitle, { color: theme.textSecondary }]}>Crea un nuevo proyecto de IA</Text>
      </View>

      <View style={styles.form}>
        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>
            Descripción *{' '}
            <Text style={[styles.required, { color: theme.textSecondary }]}>
              ({formData.description.length}/2000)
            </Text>
          </Text>
          <TextInput
            style={[
              styles.input,
              styles.textArea,
              {
                backgroundColor: theme.surface,
                borderColor: errors.description ? theme.error : theme.border,
                color: theme.text,
              },
              errors.description && styles.inputError,
            ]}
            placeholder="Describe el proyecto que quieres generar..."
            placeholderTextColor={theme.textTertiary}
            value={formData.description}
            onChangeText={(text) => handleFieldChange('description', text)}
            multiline
            numberOfLines={6}
            textAlignVertical="top"
            maxLength={2000}
          />
          {errors.description && (
            <Text style={[styles.errorText, { color: theme.error }]}>{errors.description}</Text>
          )}
        </View>

        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>Nombre del Proyecto</Text>
          <TextInput
            style={[
              styles.input,
              {
                backgroundColor: theme.surface,
                borderColor: errors.project_name ? theme.error : theme.border,
                color: theme.text,
              },
              errors.project_name && styles.inputError,
            ]}
            placeholder="project_name (opcional)"
            placeholderTextColor={theme.textTertiary}
            value={formData.project_name}
            onChangeText={(text) => handleFieldChange('project_name', text)}
            autoCapitalize="none"
            maxLength={50}
          />
          {errors.project_name && (
            <Text style={[styles.errorText, { color: theme.error }]}>{errors.project_name}</Text>
          )}
        </View>

        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>Autor</Text>
          <TextInput
            style={[styles.input, { backgroundColor: theme.surface, borderColor: theme.border, color: theme.text }]}
            placeholder="Autor"
            placeholderTextColor={theme.textTertiary}
            value={formData.author}
            onChangeText={(text) => handleFieldChange('author', text)}
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>Versión</Text>
          <TextInput
            style={[
              styles.input,
              {
                backgroundColor: theme.surface,
                borderColor: errors.version ? theme.error : theme.border,
                color: theme.text,
              },
              errors.version && styles.inputError,
            ]}
            placeholder="1.0.0"
            placeholderTextColor={theme.textTertiary}
            value={formData.version}
            onChangeText={(text) => handleFieldChange('version', text)}
            autoCapitalize="none"
          />
          {errors.version && (
            <Text style={[styles.errorText, { color: theme.error }]}>{errors.version}</Text>
          )}
        </View>

        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>Backend Framework</Text>
          <TextInput
            style={[styles.input, { backgroundColor: theme.surface, borderColor: theme.border, color: theme.text }]}
            placeholder="fastapi"
            placeholderTextColor={theme.textTertiary}
            value={formData.backend_framework}
            onChangeText={(text) => handleFieldChange('backend_framework', text)}
            autoCapitalize="none"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={[styles.label, { color: theme.text }]}>Frontend Framework</Text>
          <TextInput
            style={[styles.input, { backgroundColor: theme.surface, borderColor: theme.border, color: theme.text }]}
            placeholder="react"
            placeholderTextColor={theme.textTertiary}
            value={formData.frontend_framework}
            onChangeText={(text) => handleFieldChange('frontend_framework', text)}
            autoCapitalize="none"
          />
        </View>

        <View style={[styles.optionsSection, { backgroundColor: theme.surfaceVariant }]}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Opciones</Text>

          <View style={styles.switchGroup}>
            <View style={styles.switchLabelContainer}>
              <Text style={[styles.switchLabel, { color: theme.text }]}>Incluir Tests</Text>
              <Text style={[styles.switchDescription, { color: theme.textSecondary }]}>
                Genera tests automáticos para el proyecto
              </Text>
            </View>
            <Switch
              value={formData.generate_tests}
              onValueChange={(value) => handleFieldChange('generate_tests', value)}
              trackColor={{ false: theme.border, true: theme.primary }}
              thumbColor={theme.surface}
            />
          </View>

          <View style={styles.switchGroup}>
            <View style={styles.switchLabelContainer}>
              <Text style={[styles.switchLabel, { color: theme.text }]}>Incluir Docker</Text>
              <Text style={[styles.switchDescription, { color: theme.textSecondary }]}>
                Agrega configuración Docker y docker-compose
              </Text>
            </View>
            <Switch
              value={formData.include_docker}
              onValueChange={(value) => handleFieldChange('include_docker', value)}
              trackColor={{ false: theme.border, true: theme.primary }}
              thumbColor={theme.surface}
            />
          </View>

          <View style={styles.switchGroup}>
            <View style={styles.switchLabelContainer}>
              <Text style={[styles.switchLabel, { color: theme.text }]}>Incluir Documentación</Text>
              <Text style={[styles.switchDescription, { color: theme.textSecondary }]}>
                Genera documentación automática (README, etc.)
              </Text>
            </View>
            <Switch
              value={formData.include_docs}
              onValueChange={(value) => handleFieldChange('include_docs', value)}
              trackColor={{ false: theme.border, true: theme.primary }}
              thumbColor={theme.surface}
            />
          </View>
        </View>

        <TouchableOpacity
          style={[
            styles.generateButton,
            {
              backgroundColor: (generateMutation.isPending || Object.keys(errors).length > 0)
                ? theme.textTertiary
                : theme.primary,
              shadowColor: theme.shadow,
            },
            (generateMutation.isPending || Object.keys(errors).length > 0) &&
              styles.generateButtonDisabled,
          ]}
          onPress={() => {
            analytics.trackUserAction('generate_project');
            handleGenerate();
          }}
          disabled={generateMutation.isPending || Object.keys(errors).length > 0}
        >
          <Text style={[styles.generateButtonText, { color: theme.surface }]}>
            {generateMutation.isPending
              ? '⏳ Generando...'
              : '🚀 Generar Proyecto'}
          </Text>
        </TouchableOpacity>
      </View>
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
    marginBottom: spacing.xs,
  },
  subtitle: {
    ...typography.body,
  },
  form: {
    padding: spacing.lg,
  },
  inputGroup: {
    marginBottom: spacing.lg,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  required: {
    fontWeight: '400',
  },
  input: {
    borderWidth: 1,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    ...typography.body,
  },
  inputError: {
    borderWidth: 2,
  },
  textArea: {
    height: 120,
    textAlignVertical: 'top',
  },
  errorText: {
    ...typography.caption,
    marginTop: spacing.xs,
  },
  optionsSection: {
    marginTop: spacing.xl,
    marginBottom: spacing.lg,
    padding: spacing.lg,
    borderRadius: borderRadius.md,
  },
  sectionTitle: {
    ...typography.h3,
    marginBottom: spacing.md,
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
    marginBottom: spacing.xs,
  },
  switchDescription: {
    ...typography.caption,
  },
  generateButton: {
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    alignItems: 'center',
    marginTop: spacing.xl,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  generateButtonDisabled: {
    opacity: 0.6,
  },
  generateButtonText: {
    ...typography.body,
    fontWeight: '600',
  },
});
