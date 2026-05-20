import React, { memo, useState, useCallback, useMemo } from 'react';
import { View, StyleSheet } from 'react-native';
import { SmartList } from './smart-list';
import { Chip } from './chip';
import { COLORS, SPACING } from '../../constants/config';
import type { ListRenderItem } from 'react-native';

export interface FilterOption {
  id: string;
  label: string;
  value: string | number | boolean;
}

interface FilterableListProps<T> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  filters: FilterOption[];
  filterFn: (item: T, filterValue: string | number | boolean) => boolean;
  onFilterChange?: (activeFilters: string[]) => void;
  itemHeight?: number;
  estimatedItemHeight?: number;
  multiSelect?: boolean;
  emptyMessage?: string;
}

function FilterableListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  filters,
  filterFn,
  onFilterChange,
  itemHeight,
  estimatedItemHeight,
  multiSelect = true,
  emptyMessage = 'No items match the selected filters',
}: FilterableListProps<T>) {
  const [activeFilters, setActiveFilters] = useState<string[]>([]);

  const filteredData = useMemo(() => {
    if (activeFilters.length === 0) {
      return data;
    }

    return data.filter((item) => {
      if (multiSelect) {
        return activeFilters.some((filterId) => {
          const filter = filters.find((f) => f.id === filterId);
          return filter ? filterFn(item, filter.value) : false;
        });
      } else {
        const filter = filters.find((f) => f.id === activeFilters[0]);
        return filter ? filterFn(item, filter.value) : true;
      }
    });
  }, [data, activeFilters, filters, filterFn, multiSelect]);

  const handleFilterToggle = useCallback(
    (filterId: string) => {
      setActiveFilters((prev) => {
        let newFilters: string[];
        
        if (multiSelect) {
          if (prev.includes(filterId)) {
            newFilters = prev.filter((id) => id !== filterId);
          } else {
            newFilters = [...prev, filterId];
          }
        } else {
          newFilters = prev.includes(filterId) ? [] : [filterId];
        }

        onFilterChange?.(newFilters);
        return newFilters;
      });
    },
    [multiSelect, onFilterChange]
  );

  return (
    <View style={styles.container}>
      {filters.length > 0 && (
        <View style={styles.filtersContainer}>
          {filters.map((filter) => (
            <Chip
              key={filter.id}
              label={filter.label}
              selected={activeFilters.includes(filter.id)}
              onPress={() => handleFilterToggle(filter.id)}
              style={styles.filterChip}
            />
          ))}
        </View>
      )}
      <SmartList
        data={filteredData}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
        itemHeight={itemHeight}
        estimatedItemHeight={estimatedItemHeight}
        emptyMessage={emptyMessage}
      />
    </View>
  );
}

export const FilterableList = memo(FilterableListComponent) as <T>(
  props: FilterableListProps<T>
) => React.ReactElement;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  filtersContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: SPACING.md,
    backgroundColor: COLORS.surface,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
    gap: SPACING.sm,
  },
  filterChip: {
    marginRight: SPACING.xs,
    marginBottom: SPACING.xs,
  },
});

