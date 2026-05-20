import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface Column<T> {
  key: keyof T;
  label: string;
  render?: (value: any, row: T) => React.ReactNode;
  sortable?: boolean;
  width?: number;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  onRowPress?: (row: T) => void;
  sortable?: boolean;
  onSort?: (key: keyof T, direction: 'asc' | 'desc') => void;
  sortKey?: keyof T;
  sortDirection?: 'asc' | 'desc';
}

function DataTable<T extends Record<string, any>>({
  data,
  columns,
  onRowPress,
  sortable = false,
  onSort,
  sortKey,
  sortDirection,
}: DataTableProps<T>) {
  const { colors } = useTheme();

  const handleSort = (key: keyof T) => {
    if (sortable && onSort) {
      const direction =
        sortKey === key && sortDirection === 'asc' ? 'desc' : 'asc';
      onSort(key, direction);
    }
  };

  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false}>
      <View style={styles.container}>
        {/* Header */}
        <View
          style={[
            styles.header,
            { backgroundColor: colors.surface, borderBottomColor: colors.border },
          ]}
        >
          {columns.map((column) => (
            <TouchableOpacity
              key={String(column.key)}
              style={[
                styles.headerCell,
                { width: column.width || 120 },
                !sortable && styles.headerCellNoSort,
              ]}
              onPress={() => handleSort(column.key)}
              disabled={!sortable || !column.sortable}
              activeOpacity={sortable && column.sortable ? 0.7 : 1}
            >
              <Text
                style={[
                  styles.headerText,
                  { color: colors.text },
                  sortKey === column.key && styles.headerTextActive,
                ]}
              >
                {column.label}
              </Text>
              {sortable && column.sortable && (
                <View style={styles.sortIcon}>
                  <Ionicons
                    name={
                      sortKey === column.key && sortDirection === 'asc'
                        ? 'chevron-up'
                        : sortKey === column.key && sortDirection === 'desc'
                        ? 'chevron-down'
                        : 'chevron-up-down'
                    }
                    size={16}
                    color={
                      sortKey === column.key ? colors.primary : colors.textSecondary
                    }
                  />
                </View>
              )}
            </TouchableOpacity>
          ))}
        </View>

        {/* Rows */}
        {data.map((row, rowIndex) => (
          <TouchableOpacity
            key={rowIndex}
            style={[
              styles.row,
              {
                backgroundColor: rowIndex % 2 === 0 ? colors.card : colors.surface,
                borderBottomColor: colors.border,
              },
            ]}
            onPress={() => onRowPress?.(row)}
            disabled={!onRowPress}
            activeOpacity={onRowPress ? 0.7 : 1}
          >
            {columns.map((column) => (
              <View
                key={String(column.key)}
                style={[styles.cell, { width: column.width || 120 }]}
              >
                {column.render ? (
                  column.render(row[column.key], row)
                ) : (
                  <Text style={[styles.cellText, { color: colors.text }]}>
                    {String(row[column.key] || '')}
                  </Text>
                )}
              </View>
            ))}
          </TouchableOpacity>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  header: {
    flexDirection: 'row',
    borderBottomWidth: 2,
    paddingVertical: 12,
  },
  headerCell: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
  },
  headerCellNoSort: {
    paddingHorizontal: 12,
  },
  headerText: {
    fontSize: 14,
    fontWeight: '600',
  },
  headerTextActive: {
    color: '#007AFF',
  },
  sortIcon: {
    marginLeft: 4,
  },
  row: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    paddingVertical: 12,
  },
  cell: {
    paddingHorizontal: 12,
    justifyContent: 'center',
  },
  cellText: {
    fontSize: 14,
  },
});

export default DataTable;

