import React, { useState, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useProjectsQuery } from '../hooks/useProjectsQuery';
import { useDebounce } from '../hooks/useDebounce';
import { Project, ProjectStatus } from '../types';
import { ProjectCard } from '../components/ProjectCard';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';
import { EmptyState } from '../components/EmptyState';
import { SearchBar } from '../components/SearchBar';
import { FilterModal, FilterState } from '../components/FilterModal';
import { ProjectCardSkeleton } from '../components/SkeletonLoader';
import { NetworkStatusBar } from '../components/NetworkStatusBar';
import { QuickSearch } from '../components/QuickSearch';
import { FavoritesFilter } from '../components/FavoritesFilter';
import { AdvancedSearch, SearchFilters } from '../components/AdvancedSearch';
import { useTheme } from '../contexts/ThemeContext';
import { useAnalytics } from '../hooks/useAnalytics';
import { storage, STORAGE_KEYS } from '../utils/storage';
import { spacing, borderRadius, typography } from '../theme/colors';

export const ProjectsScreen: React.FC = () => {
  const navigation = useNavigation();
  const { theme } = useTheme();
  const analytics = useAnalytics();
  const [searchQuery, setSearchQuery] = useState('');
  const debouncedSearchQuery = useDebounce(searchQuery, 300);
  const [filters, setFilters] = useState<FilterState>({
    status: 'all',
    sortBy: 'created_at',
    sortOrder: 'desc',
  });
  const [showFilters, setShowFilters] = useState(false);
  const [showQuickSearch, setShowQuickSearch] = useState(false);
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false);
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const [favorites, setFavorites] = useState<string[]>([]);

  React.useEffect(() => {
    analytics.trackScreenView('ProjectsScreen');
    loadFavorites();
  }, []);

  const loadFavorites = async () => {
    try {
      const favs = await storage.get<string[]>(STORAGE_KEYS.FAVORITES) || [];
      setFavorites(favs);
    } catch (error) {
      console.error('Error loading favorites:', error);
    }
  };

  const { data: projects = [], isLoading, error, refetch, isFetching } = useProjectsQuery({
    status: filters.status !== 'all' ? filters.status : undefined,
    limit: 100,
  });

  const filteredAndSortedProjects = useMemo(() => {
    let result = [...projects];

    if (showFavoritesOnly) {
      result = result.filter((p) => favorites.includes(p.project_id));
    }

    if (debouncedSearchQuery.trim()) {
      const query = debouncedSearchQuery.toLowerCase();
      result = result.filter(
        (p) =>
          p.project_name.toLowerCase().includes(query) ||
          p.description.toLowerCase().includes(query) ||
          p.author.toLowerCase().includes(query)
      );
    }

    if (filters.status !== 'all') {
      result = result.filter((p) => p.status === filters.status);
    }

    if (filters.sortBy) {
      result.sort((a, b) => {
        let comparison = 0;
        switch (filters.sortBy) {
          case 'name':
            comparison = a.project_name.localeCompare(b.project_name);
            break;
          case 'status':
            comparison = a.status.localeCompare(b.status);
            break;
          case 'created_at':
          default:
            comparison =
              new Date(a.created_at).getTime() -
              new Date(b.created_at).getTime();
            break;
        }
        return filters.sortOrder === 'asc' ? comparison : -comparison;
      });
    }

    return result;
  }, [projects, debouncedSearchQuery, filters, showFavoritesOnly, favorites]);

  const handleProjectPress = (project: Project) => {
    navigation.navigate(
      'ProjectDetail' as never,
      { projectId: project.project_id } as never
    );
  };

  const handleApplyFilters = (newFilters: FilterState) => {
    setFilters(newFilters);
    analytics.trackUserAction('apply_filters', { filters: newFilters });
  };

  const handleAdvancedSearchApply = (searchFilters: SearchFilters) => {
    setFilters({
      status: searchFilters.status === 'all' ? 'all' : searchFilters.status as ProjectStatus,
      sortBy: searchFilters.sortBy,
      sortOrder: searchFilters.sortOrder,
    });
    if (searchFilters.query) {
      setSearchQuery(searchFilters.query);
    }
    analytics.trackUserAction('advanced_search', { filters: searchFilters });
  };

  const handleQuickSearchSelect = (project: Project) => {
    navigation.navigate(
      'ProjectDetail' as never,
      { projectId: project.project_id } as never
    );
    analytics.trackUserAction('quick_search_select', { projectId: project.project_id });
  };

  if (isLoading && projects.length === 0) {
    return (
      <View style={styles.container}>
        <NetworkStatusBar />
        <View style={styles.header}>
          <Text style={styles.title}>Proyectos</Text>
        </View>
        <View style={styles.listContent}>
          {[1, 2, 3, 4, 5].map((i) => (
            <ProjectCardSkeleton key={i} />
          ))}
        </View>
      </View>
    );
  }

  if (error) {
    return (
      <>
        <NetworkStatusBar />
        <ErrorMessage error={error} onRetry={() => refetch()} />
      </>
    );
  }

  return (
    <View style={[styles.container, { backgroundColor: theme.background }]}>
      <NetworkStatusBar />
      <View style={[styles.header, { backgroundColor: theme.surface, borderBottomColor: theme.border }]}>
        <Text style={[styles.title, { color: theme.text }]}>Proyectos</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity
            style={[styles.searchButton, { backgroundColor: theme.surfaceVariant }]}
            onPress={() => setShowQuickSearch(true)}
          >
            <Text style={styles.searchButtonText}>🔍</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.filterButton, { backgroundColor: theme.surfaceVariant }]}
            onPress={() => setShowAdvancedSearch(true)}
          >
            <Text style={styles.filterButtonText}>🔽</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.addButton, { backgroundColor: theme.primary }]}
            onPress={() => navigation.navigate('Generate' as never)}
          >
            <Text style={[styles.addButtonText, { color: theme.surface }]}>➕</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={[styles.searchContainer, { backgroundColor: theme.surface }]}>
        <View style={styles.searchRow}>
          <View style={styles.searchBarContainer}>
            <SearchBar
              value={searchQuery}
              onChangeText={setSearchQuery}
              placeholder="Buscar proyectos..."
              onClear={() => setSearchQuery('')}
            />
          </View>
          <FavoritesFilter
            onToggle={(show) => {
              setShowFavoritesOnly(show);
              analytics.trackUserAction('toggle_favorites_filter', { enabled: show });
            }}
            initialValue={showFavoritesOnly}
          />
        </View>
      </View>

      {filteredAndSortedProjects.length === 0 ? (
        <EmptyState
          message={
            searchQuery || filters.status !== 'all'
              ? 'No se encontraron proyectos con los filtros aplicados'
              : 'No hay proyectos aún. ¡Crea tu primer proyecto!'
          }
        />
      ) : (
        <FlatList
          data={filteredAndSortedProjects}
          keyExtractor={(item) => item.project_id}
          renderItem={({ item }) => (
            <ProjectCard
              project={item}
              onPress={() => handleProjectPress(item)}
            />
          )}
          contentContainerStyle={styles.listContent}
          refreshing={isFetching}
          onRefresh={() => refetch()}
          ListFooterComponent={
            isFetching && projects.length > 0 ? (
              <View style={styles.footerLoader}>
                <LoadingSpinner message="" size="small" />
              </View>
            ) : null
          }
        />
      )}

      <FilterModal
        visible={showFilters}
        onClose={() => setShowFilters(false)}
        onApply={handleApplyFilters}
        initialFilters={filters}
      />

      <QuickSearch
        visible={showQuickSearch}
        onClose={() => setShowQuickSearch(false)}
        onSelect={handleQuickSearchSelect}
        projects={projects}
      />

      <AdvancedSearch
        visible={showAdvancedSearch}
        onClose={() => setShowAdvancedSearch(false)}
        onApply={handleAdvancedSearchApply}
        initialFilters={{
          query: searchQuery,
          status: filters.status === 'all' ? 'all' : filters.status,
          sortBy: filters.sortBy,
          sortOrder: filters.sortOrder,
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.lg,
    borderBottomWidth: 1,
  },
  title: {
    ...typography.h2,
  },
  headerActions: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  searchButton: {
    width: 44,
    height: 44,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
  },
  searchButtonText: {
    fontSize: 20,
  },
  filterButton: {
    width: 44,
    height: 44,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
  },
  filterButtonText: {
    fontSize: 20,
  },
  addButton: {
    width: 44,
    height: 44,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
  },
  addButtonText: {
    fontSize: 24,
  },
  searchContainer: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.md,
  },
  searchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  searchBarContainer: {
    flex: 1,
  },
  listContent: {
    padding: spacing.lg,
  },
  footerLoader: {
    padding: spacing.lg,
  },
});
