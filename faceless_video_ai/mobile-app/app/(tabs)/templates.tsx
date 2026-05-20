import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useTemplates } from '@/hooks/use-templates';
import { Loading } from '@/components/ui/loading';

export default function TemplatesScreen() {
  const router = useRouter();
  const { data, isLoading, refetch, isRefetching } = useTemplates();

  const renderTemplate = ({ item: template }: { item: { name: string; description?: string } }) => (
    <TouchableOpacity
      style={styles.templateCard}
      onPress={() => router.push(`/template-detail?templateName=${template.name}`)}
    >
      <Text style={styles.templateName}>{template.name}</Text>
      {template.description && (
        <Text style={styles.templateDescription} numberOfLines={2}>
          {template.description}
        </Text>
      )}
    </TouchableOpacity>
  );

  if (isLoading) {
    return <Loading message="Loading templates..." />;
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <Text style={styles.title}>Templates</Text>
      </View>

      {data?.templates && data.templates.length > 0 ? (
        <FlatList
          data={data.templates}
          renderItem={renderTemplate}
          keyExtractor={(item) => item.name}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={isRefetching}
              onRefresh={() => refetch()}
            />
          }
        />
      ) : (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No templates available</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  header: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#000000',
  },
  listContent: {
    padding: 20,
  },
  templateCard: {
    backgroundColor: '#F5F5F5',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  templateName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000000',
    marginBottom: 8,
  },
  templateDescription: {
    fontSize: 14,
    color: '#666666',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 16,
    color: '#666666',
  },
});


