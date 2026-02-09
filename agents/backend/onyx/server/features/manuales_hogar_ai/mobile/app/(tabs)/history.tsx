import { View, StyleSheet, FlatList, RefreshControl } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery } from '@tanstack/react-query';
import { useApp } from '@/lib/context/app-context';
import { manualService } from '@/services/api/manual-service';
import { ManualListItem } from '@/components/history/manual-list-item';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { ErrorMessage } from '@/components/ui/error-message';
import { EmptyState } from '@/components/ui/empty-state';

export default function HistoryScreen() {
  const { state } = useApp();
  const colors = state.colors;

  const {
    data: manualsData,
    isLoading,
    error,
    refetch,
    isRefetching,
  } = useQuery({
    queryKey: ['manuals', 'all'],
    queryFn: () => manualService.getManuals({ limit: 50 }),
  });

  if (isLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
        <LoadingSpinner />
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
        <ErrorMessage message="Error al cargar el historial" onRetry={() => refetch()} />
      </SafeAreaView>
    );
  }

  const manuals = manualsData?.manuals || [];

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
      <FlatList
        data={manuals}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => <ManualListItem manual={item} />}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl refreshing={isRefetching} onRefresh={refetch} tintColor={colors.tint} />
        }
        ListEmptyComponent={
          <EmptyState
            icon="document-text-outline"
            title="No hay manuales"
            message="Comienza generando tu primer manual"
          />
        }
        ItemSeparatorComponent={() => <View style={[styles.separator, { backgroundColor: colors.border }]} />}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  listContent: {
    padding: 20,
  },
  separator: {
    height: 1,
    marginVertical: 8,
  },
});




