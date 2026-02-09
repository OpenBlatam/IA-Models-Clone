import React, { useState, useMemo } from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { SearchInput } from './SearchInput';
import { spacing, typography } from '../theme/colors';

interface SearchableListProps<T> {
  data: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  searchKeys?: (keyof T)[];
  placeholder?: string;
  emptyMessage?: string;
  ListHeaderComponent?: React.ReactNode;
  ListFooterComponent?: React.ReactNode;
  keyExtractor?: (item: T, index: number) => string;
}

export function SearchableList<T extends Record<string, any>>({
  data,
  renderItem,
  searchKeys,
  placeholder = 'Buscar...',
  emptyMessage = 'No se encontraron resultados',
  ListHeaderComponent,
  ListFooterComponent,
  keyExtractor = (item, index) => item.id?.toString() || index.toString(),
}: SearchableListProps<T>) {
  const { theme } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');

  const filteredData = useMemo(() => {
    if (!searchQuery.trim()) return data;

    const query = searchQuery.toLowerCase();
    const keys = searchKeys || (Object.keys(data[0] || {}) as (keyof T)[]);

    return data.filter((item) => {
      return keys.some((key) => {
        const value = item[key];
        return value?.toString().toLowerCase().includes(query);
      });
    });
  }, [data, searchQuery, searchKeys]);

  return (
    <View style={[styles.container, { backgroundColor: theme.background }]}>
      <View style={[styles.searchContainer, { backgroundColor: theme.surface }]}>
        <SearchInput
          placeholder={placeholder}
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
      </View>
      <FlatList
        data={filteredData}
        renderItem={({ item, index }) => renderItem(item, index)}
        keyExtractor={keyExtractor}
        ListHeaderComponent={ListHeaderComponent}
        ListFooterComponent={ListFooterComponent}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={[styles.emptyText, { color: theme.textSecondary }]}>
              {emptyMessage}
            </Text>
          </View>
        }
        contentContainerStyle={styles.listContent}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  searchContainer: {
    padding: spacing.md,
  },
  listContent: {
    flexGrow: 1,
  },
  emptyContainer: {
    padding: spacing.xl,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyText: {
    ...typography.body,
  },
});

