import { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  TextInput,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useTemplates, useCreateTemplate, useDeleteTemplate } from '@/hooks/useApi';
import { ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { Template } from '@/types';
import { EmptyState } from '@/components/ui/EmptyState';

export default function TemplatesScreen() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const { data: templates, isLoading, refetch } = useTemplates({ query: searchQuery || undefined });
  const createTemplate = useCreateTemplate();
  const deleteTemplate = useDeleteTemplate();

  const handleDelete = (templateId: string, templateName: string) => {
    Alert.alert('Delete Template', `Are you sure you want to delete "${templateName}"?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            await deleteTemplate.mutateAsync(templateId);
            Alert.alert('Success', 'Template deleted');
            refetch();
          } catch (error) {
            Alert.alert('Error', 'Failed to delete template');
          }
        },
      },
    ]);
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <View style={styles.searchContainer}>
          <Ionicons name="search" size={20} color="#9ca3af" style={styles.searchIcon} />
          <TextInput
            style={styles.searchInput}
            placeholder="Search templates..."
            placeholderTextColor="#9ca3af"
            value={searchQuery}
            onChangeText={setSearchQuery}
          />
        </View>
        <TouchableOpacity
          style={styles.addButton}
          onPress={() => router.push('/templates/create')}
        >
          <Ionicons name="add" size={24} color="#fff" />
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refetch} />}
      >
        {isLoading && !templates ? (
          <ActivityIndicator size="large" color="#0ea5e9" style={styles.loader} />
        ) : templates && templates.length > 0 ? (
          <View style={styles.templatesList}>
            {templates.map((template) => (
              <TemplateCard
                key={template.template_id}
                template={template}
                onDelete={() => handleDelete(template.template_id, template.name)}
              />
            ))}
          </View>
        ) : (
          <EmptyState
            icon="document-outline"
            title="No templates found"
            message="Create your first template to reuse content"
            actionLabel="Create Template"
            onAction={() => router.push('/templates/create')}
          />
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function TemplateCard({ template, onDelete }: { template: Template; onDelete: () => void }) {
  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={styles.cardHeaderLeft}>
          <Ionicons name="document-text" size={24} color="#0ea5e9" />
          <View style={styles.cardTitleContainer}>
            <Text style={styles.cardTitle}>{template.name}</Text>
            {template.platform && (
              <View style={styles.platformTag}>
                <Text style={styles.platformText}>{template.platform}</Text>
              </View>
            )}
          </View>
        </View>
        <TouchableOpacity onPress={onDelete}>
          <Ionicons name="trash" size={20} color="#ef4444" />
        </TouchableOpacity>
      </View>

      <Text style={styles.cardContent} numberOfLines={3}>
        {template.content}
      </Text>

      {template.variables && template.variables.length > 0 && (
        <View style={styles.variables}>
          <Text style={styles.variablesLabel}>Variables:</Text>
          <View style={styles.variablesList}>
            {template.variables.map((variable) => (
              <View key={variable} style={styles.variableTag}>
                <Text style={styles.variableText}>{variable}</Text>
              </View>
            ))}
          </View>
        </View>)}

      {template.category && (
        <View style={styles.category}>
          <Text style={styles.categoryText}>{template.category}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    paddingHorizontal: 12,
    marginBottom: 12,
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    height: 40,
    fontSize: 14,
    color: '#1f2937',
  },
  addButton: {
    backgroundColor: '#0ea5e9',
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'flex-end',
  },
  scrollView: {
    flex: 1,
  },
  loader: {
    marginTop: 40,
  },
  templatesList: {
    padding: 16,
    gap: 16,
  },
  card: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  cardHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  cardTitleContainer: {
    marginLeft: 12,
    flex: 1,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  platformTag: {
    alignSelf: 'flex-start',
    backgroundColor: '#e0f2fe',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  platformText: {
    fontSize: 10,
    color: '#0369a1',
    fontWeight: '500',
  },
  cardContent: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
    marginBottom: 12,
  },
  variables: {
    marginTop: 8,
  },
  variablesLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 6,
  },
  variablesList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  variableTag: {
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  variableText: {
    fontSize: 11,
    color: '#4b5563',
  },
  category: {
    marginTop: 8,
    alignSelf: 'flex-start',
  },
  categoryText: {
    fontSize: 12,
    color: '#6b7280',
    fontStyle: 'italic',
  },
});

