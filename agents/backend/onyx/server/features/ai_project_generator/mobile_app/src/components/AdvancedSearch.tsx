import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Modal,
  ScrollView,
} from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';
import { ProjectStatus } from '../types';

interface AdvancedSearchProps {
  visible: boolean;
  onClose: () => void;
  onApply: (filters: SearchFilters) => void;
  initialFilters?: SearchFilters;
}

export interface SearchFilters {
  query: string;
  status?: ProjectStatus | 'all';
  author?: string;
  dateFrom?: string;
  dateTo?: string;
  favoritesOnly?: boolean;
  sortBy: 'name' | 'status' | 'created_at' | 'author';
  sortOrder: 'asc' | 'desc';
}

export const AdvancedSearch: React.FC<AdvancedSearchProps> = ({
  visible,
  onClose,
  onApply,
  initialFilters,
}) => {
  const { theme } = useTheme();
  const [filters, setFilters] = useState<SearchFilters>(
    initialFilters || {
      query: '',
      status: 'all',
      author: '',
      dateFrom: '',
      dateTo: '',
      favoritesOnly: false,
      sortBy: 'created_at',
      sortOrder: 'desc',
    }
  );

  const handleApply = () => {
    hapticFeedback.selection();
    onApply(filters);
    onClose();
  };

  const handleReset = () => {
    hapticFeedback.light();
    setFilters({
      query: '',
      status: 'all',
      author: '',
      dateFrom: '',
      dateTo: '',
      favoritesOnly: false,
      sortBy: 'created_at',
      sortOrder: 'desc',
    });
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={[styles.overlay, { backgroundColor: 'rgba(0,0,0,0.5)' }]}>
        <View style={[styles.container, { backgroundColor: theme.surface }]}>
          <View style={[styles.header, { borderBottomColor: theme.border }]}>
            <Text style={[styles.title, { color: theme.text }]}>Búsqueda Avanzada</Text>
            <TouchableOpacity onPress={onClose}>
              <Text style={[styles.closeButton, { color: theme.primary }]}>✕</Text>
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.content}>
            <View style={styles.section}>
              <Text style={[styles.label, { color: theme.text }]}>Buscar texto</Text>
              <TextInput
                style={[
                  styles.input,
                  {
                    backgroundColor: theme.surfaceVariant,
                    borderColor: theme.border,
                    color: theme.text,
                  },
                ]}
                placeholder="Nombre, descripción..."
                placeholderTextColor={theme.textTertiary}
                value={filters.query}
                onChangeText={(text) => setFilters({ ...filters, query: text })}
              />
            </View>

            <View style={styles.section}>
              <Text style={[styles.label, { color: theme.text }]}>Autor</Text>
              <TextInput
                style={[
                  styles.input,
                  {
                    backgroundColor: theme.surfaceVariant,
                    borderColor: theme.border,
                    color: theme.text,
                  },
                ]}
                placeholder="Filtrar por autor..."
                placeholderTextColor={theme.textTertiary}
                value={filters.author}
                onChangeText={(text) => setFilters({ ...filters, author: text })}
              />
            </View>

            <View style={styles.section}>
              <Text style={[styles.label, { color: theme.text }]}>Estado</Text>
              <View style={styles.statusContainer}>
                {(['all', 'queued', 'processing', 'completed', 'failed', 'cancelled'] as const).map(
                  (status) => (
                    <TouchableOpacity
                      key={status}
                      style={[
                        styles.statusButton,
                        {
                          backgroundColor:
                            filters.status === status ? theme.primary : theme.surfaceVariant,
                          borderColor: theme.border,
                        },
                      ]}
                      onPress={() => {
                        hapticFeedback.selection();
                        setFilters({ ...filters, status });
                      }}
                    >
                      <Text
                        style={[
                          styles.statusButtonText,
                          {
                            color:
                              filters.status === status ? theme.surface : theme.text,
                          },
                        ]}
                      >
                        {status === 'all' ? 'Todos' : status}
                      </Text>
                    </TouchableOpacity>
                  )
                )}
              </View>
            </View>

            <View style={styles.section}>
              <Text style={[styles.label, { color: theme.text }]}>Ordenar por</Text>
              <View style={styles.sortContainer}>
                {(['name', 'status', 'created_at', 'author'] as const).map((sortBy) => (
                  <TouchableOpacity
                    key={sortBy}
                    style={[
                      styles.sortButton,
                      {
                        backgroundColor:
                          filters.sortBy === sortBy ? theme.primary : theme.surfaceVariant,
                        borderColor: theme.border,
                      },
                    ]}
                    onPress={() => {
                      hapticFeedback.selection();
                      setFilters({ ...filters, sortBy });
                    }}
                  >
                    <Text
                      style={[
                        styles.sortButtonText,
                        {
                          color: filters.sortBy === sortBy ? theme.surface : theme.text,
                        },
                      ]}
                    >
                      {sortBy === 'created_at' ? 'Fecha' : sortBy}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            <View style={styles.section}>
              <Text style={[styles.label, { color: theme.text }]}>Orden</Text>
              <View style={styles.orderContainer}>
                <TouchableOpacity
                  style={[
                    styles.orderButton,
                    {
                      backgroundColor:
                        filters.sortOrder === 'asc' ? theme.primary : theme.surfaceVariant,
                      borderColor: theme.border,
                    },
                  ]}
                  onPress={() => {
                    hapticFeedback.selection();
                    setFilters({ ...filters, sortOrder: 'asc' });
                  }}
                >
                  <Text
                    style={[
                      styles.orderButtonText,
                      {
                        color: filters.sortOrder === 'asc' ? theme.surface : theme.text,
                      },
                    ]}
                  >
                    ↑ Ascendente
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[
                    styles.orderButton,
                    {
                      backgroundColor:
                        filters.sortOrder === 'desc' ? theme.primary : theme.surfaceVariant,
                      borderColor: theme.border,
                    },
                  ]}
                  onPress={() => {
                    hapticFeedback.selection();
                    setFilters({ ...filters, sortOrder: 'desc' });
                  }}
                >
                  <Text
                    style={[
                      styles.orderButtonText,
                      {
                        color: filters.sortOrder === 'desc' ? theme.surface : theme.text,
                      },
                    ]}
                  >
                    ↓ Descendente
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </ScrollView>

          <View style={[styles.footer, { borderTopColor: theme.border }]}>
            <TouchableOpacity
              style={[styles.resetButton, { backgroundColor: theme.surfaceVariant }]}
              onPress={handleReset}
            >
              <Text style={[styles.resetButtonText, { color: theme.text }]}>Limpiar</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.applyButton, { backgroundColor: theme.primary }]}
              onPress={handleApply}
            >
              <Text style={[styles.applyButtonText, { color: theme.surface }]}>Aplicar</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  container: {
    borderTopLeftRadius: borderRadius.xl,
    borderTopRightRadius: borderRadius.xl,
    maxHeight: '90%',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.xl,
    borderBottomWidth: 1,
  },
  title: {
    ...typography.h2,
  },
  closeButton: {
    fontSize: 24,
    fontWeight: '600',
  },
  content: {
    padding: spacing.xl,
  },
  section: {
    marginBottom: spacing.xl,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  input: {
    borderWidth: 1,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    ...typography.body,
  },
  statusContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  statusButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    borderWidth: 1,
  },
  statusButtonText: {
    ...typography.caption,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  sortContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  sortButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    borderWidth: 1,
  },
  sortButtonText: {
    ...typography.caption,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  orderContainer: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  orderButton: {
    flex: 1,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    alignItems: 'center',
  },
  orderButtonText: {
    ...typography.bodySmall,
    fontWeight: '600',
  },
  footer: {
    flexDirection: 'row',
    padding: spacing.xl,
    gap: spacing.md,
    borderTopWidth: 1,
  },
  resetButton: {
    flex: 1,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
  },
  resetButtonText: {
    ...typography.body,
    fontWeight: '600',
  },
  applyButton: {
    flex: 1,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
  },
  applyButtonText: {
    ...typography.body,
    fontWeight: '600',
  },
});

