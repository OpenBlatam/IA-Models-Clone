import React, { memo, useState, useCallback, useMemo } from 'react';
import { View, StyleSheet } from 'react-native';
import { SmartList } from './smart-list';
import { SearchBar } from './search-bar';
import { useDebouncedList } from '../../hooks/use-debounced-list';
import { COLORS, SPACING } from '../../constants/config';
import type { ListRenderItem } from 'react-native';

interface SearchableListProps<T> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  searchPlaceholder?: string;
  searchFn?: (item: T, query: string) => boolean;
  filterFn?: (item: T) => boolean;
  sortFn?: (a: T, b: T) => number;
  itemHeight?: number;
  estimatedItemHeight?: number;
  showSearchBar?: boolean;
  emptyMessage?: string;
}

function SearchableListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  searchPlaceholder = 'Search...',
  searchFn,
  filterFn,
  sortFn,
  itemHeight,
  estimatedItemHeight,
  showSearchBar = true,
  emptyMessage = 'No items found',
  ...listProps
}: SearchableListProps<T>) {
  const [searchQuery, setSearchQuery] = useState('');

  const { filteredItems, stats } = useDebouncedList({
    items: data,
    searchQuery,
    searchFn,
    filterFn,
    sortFn,
  });

  const handleSearchChange = useCallback((query: string) => {
    setSearchQuery(query);
  }, []);

  const handleClearSearch = useCallback(() => {
    setSearchQuery('');
  }, []);

  return (
    <View style={styles.container}>
      {showSearchBar && (
        <View style={styles.searchContainer}>
          <SearchBar
            value={searchQuery}
            onChangeText={handleSearchChange}
            placeholder={searchPlaceholder}
            onClear={handleClearSearch}
            showClearButton
          />
          {stats.isFiltered && (
            <View style={styles.statsContainer}>
              {/* Stats can be displayed here if needed */}
            </View>
          )}
        </View>
      )}
      <SmartList
        data={filteredItems}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
        itemHeight={itemHeight}
        estimatedItemHeight={estimatedItemHeight}
        emptyMessage={emptyMessage}
        {...listProps}
      />
    </View>
  );
}

export const SearchableList = memo(SearchableListComponent) as <T>(
  props: SearchableListProps<T>
) => React.ReactElement;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  searchContainer: {
    padding: SPACING.md,
    backgroundColor: COLORS.surface,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  statsContainer: {
    marginTop: SPACING.xs,
  },
});

