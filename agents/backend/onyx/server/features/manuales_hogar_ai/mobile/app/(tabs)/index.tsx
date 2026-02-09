import { View, Text, StyleSheet, ScrollView, RefreshControl } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery } from '@tanstack/react-query';
import { useColorScheme } from 'react-native';
import { Colors } from '@/constants/colors';
import { HomeHeader } from '@/components/home/home-header';
import { QuickActions } from '@/components/home/quick-actions';
import { RecentManuals } from '@/components/home/recent-manuals';
import { CategoryGrid } from '@/components/home/category-grid';
import { FeaturedCategories } from '@/components/home/featured-categories';
import { manualService } from '@/services/api/manual-service';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { ErrorMessage } from '@/components/ui/error-message';

export default function HomeScreen() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const colors = isDark ? Colors.dark : Colors.light;

  const {
    data: recentManuals,
    isLoading,
    error,
    refetch,
    isRefetching,
  } = useQuery({
    queryKey: ['manuals', 'recent'],
    queryFn: () => manualService.getRecentManuals({ limit: 10 }),
  });

  const {
    data: categories,
    isLoading: categoriesLoading,
  } = useQuery({
    queryKey: ['categories'],
    queryFn: () => manualService.getCategories(),
  });

  if (isLoading || categoriesLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
        <LoadingSpinner />
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
        <ErrorMessage message="Error al cargar los datos" onRetry={() => refetch()} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl refreshing={isRefetching} onRefresh={refetch} tintColor={colors.tint} />
        }
        showsVerticalScrollIndicator={false}
      >
        <HomeHeader />
        <QuickActions />
        <FeaturedCategories />
        {categories && <CategoryGrid categories={categories} />}
        {recentManuals && <RecentManuals manuals={recentManuals} />}
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
    paddingBottom: 20,
  },
});


