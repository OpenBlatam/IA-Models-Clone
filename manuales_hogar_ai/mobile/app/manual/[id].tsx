import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { useApp } from '@/lib/context/app-context';
import { manualService } from '@/services/api/manual-service';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { ErrorMessage } from '@/components/ui/error-message';
import { ManualDetailContent } from '@/components/manual/manual-detail-content';

export default function ManualDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { state } = useApp();
  const colors = state.colors;

  const { data: manual, isLoading, error } = useQuery({
    queryKey: ['manual', id],
    queryFn: () => manualService.getManualById(id),
    enabled: !!id,
  });

  if (isLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
        <LoadingSpinner />
      </SafeAreaView>
    );
  }

  if (error || !manual) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
        <ErrorMessage message="Error al cargar el manual" />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <ManualDetailContent manual={manual} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
  },
});




