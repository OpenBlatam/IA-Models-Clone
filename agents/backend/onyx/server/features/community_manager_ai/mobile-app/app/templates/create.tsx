import { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useCreateTemplate } from '@/hooks/useApi';
import { templateSchema, type TemplateFormData } from '@/lib/validation';
import { Input } from '@/components/ui/Input';
import { TextArea } from '@/components/ui/TextArea';
import { Button } from '@/components/ui/Button';
import { PLATFORMS } from '@/utils/constants';
import { Ionicons } from '@expo/vector-icons';
import { showToast } from '@/utils/toast';

export default function CreateTemplateScreen() {
  const router = useRouter();
  const createTemplate = useCreateTemplate();

  const {
    control,
    handleSubmit,
    formState: { errors, isSubmitting },
    watch,
    setValue,
  } = useForm<TemplateFormData>({
    resolver: zodResolver(templateSchema),
    defaultValues: {
      name: '',
      content: '',
      variables: [],
    },
  });

  const content = watch('content');

  // Extract variables from content (e.g., {variable})
  const extractVariables = (text: string): string[] => {
    const regex = /\{(\w+)\}/g;
    const matches = text.match(regex);
    if (!matches) return [];
    return [...new Set(matches.map((match) => match.slice(1, -1)))];
  };

  const variables = extractVariables(content || '');

  const onSubmit = async (data: TemplateFormData) => {
    try {
      const templateData = {
        ...data,
        variables: variables.length > 0 ? variables : undefined,
      };
      await createTemplate.mutateAsync(templateData);
      showToast.success('Template created successfully');
      router.back();
    } catch (error: any) {
      showToast.error(error.message || 'Failed to create template');
    }
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="arrow-back" size={24} color="#1f2937" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Create Template</Text>
          <View style={{ width: 24 }} />
        </View>

        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          <View style={styles.content}>
            <Controller
              control={control}
              name="name"
              render={({ field: { onChange, onBlur, value } }) => (
                <Input
                  label="Template Name *"
                  placeholder="Enter template name"
                  value={value}
                  onChangeText={onChange}
                  onBlur={onBlur}
                  error={errors.name?.message}
                />
              )}
            />

            <Controller
              control={control}
              name="content"
              render={({ field: { onChange, onBlur, value } }) => (
                <TextArea
                  label="Content *"
                  placeholder="Use {variable} for variables. Example: Hello {name}!"
                  value={value}
                  onChangeText={onChange}
                  onBlur={onBlur}
                  error={errors.content?.message}
                  rows={8}
                  helperText="Use {variableName} to create variables"
                />
              )}
            />

            {variables.length > 0 && (
              <View style={styles.variablesSection}>
                <Text style={styles.sectionTitle}>Detected Variables</Text>
                <View style={styles.variablesList}>
                  {variables.map((variable) => (
                    <View key={variable} style={styles.variableTag}>
                      <Text style={styles.variableText}>{`{${variable}}`}</Text>
                    </View>
                  ))}
                </View>
              </View>
            )}

            <Controller
              control={control}
              name="category"
              render={({ field: { onChange, onBlur, value } }) => (
                <Input
                  label="Category (Optional)"
                  placeholder="e.g., Marketing, Announcement"
                  value={value || ''}
                  onChangeText={onChange}
                  onBlur={onBlur}
                  error={errors.category?.message}
                />
              )}
            />
          </View>
        </ScrollView>

        <View style={styles.footer}>
          <Button
            title="Create Template"
            onPress={handleSubmit(onSubmit)}
            loading={isSubmitting}
            disabled={isSubmitting}
            style={styles.submitButton}
          />
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  keyboardView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
  },
  variablesSection: {
    marginBottom: 24,
    padding: 16,
    backgroundColor: '#f0f9ff',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#bae6fd',
  },
  variablesList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  variableTag: {
    backgroundColor: '#0ea5e9',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  variableText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
  footer: {
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  submitButton: {
    width: '100%',
  },
});

