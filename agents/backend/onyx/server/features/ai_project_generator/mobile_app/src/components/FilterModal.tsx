import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { colors, spacing, borderRadius, typography } from '../theme/colors';
import { ProjectStatus } from '../types';

interface FilterModalProps {
  visible: boolean;
  onClose: () => void;
  onApply: (filters: FilterState) => void;
  initialFilters?: FilterState;
}

export interface FilterState {
  status?: ProjectStatus | 'all';
  author?: string;
  sortBy?: 'created_at' | 'name' | 'status';
  sortOrder?: 'asc' | 'desc';
}

export const FilterModal: React.FC<FilterModalProps> = ({
  visible,
  onClose,
  onApply,
  initialFilters = {},
}) => {
  const [filters, setFilters] = useState<FilterState>({
    status: 'all',
    sortBy: 'created_at',
    sortOrder: 'desc',
    ...initialFilters,
  });

  const statusOptions: Array<ProjectStatus | 'all'> = [
    'all',
    ProjectStatus.QUEUED,
    ProjectStatus.PROCESSING,
    ProjectStatus.COMPLETED,
    ProjectStatus.FAILED,
    ProjectStatus.CANCELLED,
  ];

  const sortOptions: Array<'created_at' | 'name' | 'status'> = [
    'created_at',
    'name',
    'status',
  ];

  const handleApply = () => {
    onApply(filters);
    onClose();
  };

  const handleReset = () => {
    const resetFilters: FilterState = {
      status: 'all',
      sortBy: 'created_at',
      sortOrder: 'desc',
    };
    setFilters(resetFilters);
    onApply(resetFilters);
    onClose();
  };

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent={true}
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <View style={styles.modal}>
          <View style={styles.header}>
            <Text style={styles.title}>Filtros</Text>
            <TouchableOpacity onPress={onClose}>
              <Text style={styles.closeButton}>✕</Text>
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.content}>
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Estado</Text>
              <View style={styles.optionsContainer}>
                {statusOptions.map((status) => (
                  <TouchableOpacity
                    key={status}
                    style={[
                      styles.option,
                      filters.status === status && styles.optionSelected,
                    ]}
                    onPress={() => setFilters({ ...filters, status })}
                  >
                    <Text
                      style={[
                        styles.optionText,
                        filters.status === status && styles.optionTextSelected,
                      ]}
                    >
                      {status === 'all' ? 'Todos' : status.toUpperCase()}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Ordenar por</Text>
              <View style={styles.optionsContainer}>
                {sortOptions.map((sort) => (
                  <TouchableOpacity
                    key={sort}
                    style={[
                      styles.option,
                      filters.sortBy === sort && styles.optionSelected,
                    ]}
                    onPress={() => setFilters({ ...filters, sortBy: sort })}
                  >
                    <Text
                      style={[
                        styles.optionText,
                        filters.sortBy === sort && styles.optionTextSelected,
                      ]}
                    >
                      {sort === 'created_at'
                        ? 'Fecha'
                        : sort === 'name'
                        ? 'Nombre'
                        : 'Estado'}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Orden</Text>
              <View style={styles.optionsContainer}>
                <TouchableOpacity
                  style={[
                    styles.option,
                    filters.sortOrder === 'asc' && styles.optionSelected,
                  ]}
                  onPress={() => setFilters({ ...filters, sortOrder: 'asc' })}
                >
                  <Text
                    style={[
                      styles.optionText,
                      filters.sortOrder === 'asc' && styles.optionTextSelected,
                    ]}
                  >
                    Ascendente
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[
                    styles.option,
                    filters.sortOrder === 'desc' && styles.optionSelected,
                  ]}
                  onPress={() => setFilters({ ...filters, sortOrder: 'desc' })}
                >
                  <Text
                    style={[
                      styles.optionText,
                      filters.sortOrder === 'desc' && styles.optionTextSelected,
                    ]}
                  >
                    Descendente
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </ScrollView>

          <View style={styles.footer}>
            <TouchableOpacity
              style={[styles.button, styles.resetButton]}
              onPress={handleReset}
            >
              <Text style={styles.resetButtonText}>Resetear</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.button, styles.applyButton]}
              onPress={handleApply}
            >
              <Text style={styles.applyButtonText}>Aplicar</Text>
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
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modal: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: borderRadius.lg,
    borderTopRightRadius: borderRadius.lg,
    maxHeight: '80%',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  title: {
    ...typography.h3,
    color: colors.text,
  },
  closeButton: {
    fontSize: 24,
    color: colors.textSecondary,
  },
  content: {
    padding: spacing.lg,
  },
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    ...typography.bodySmall,
    fontWeight: '600',
    color: colors.text,
    marginBottom: spacing.md,
  },
  optionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  option: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    backgroundColor: colors.surfaceVariant,
    borderWidth: 1,
    borderColor: colors.border,
  },
  optionSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  optionText: {
    ...typography.bodySmall,
    color: colors.text,
  },
  optionTextSelected: {
    color: colors.surface,
    fontWeight: '600',
  },
  footer: {
    flexDirection: 'row',
    padding: spacing.lg,
    gap: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  button: {
    flex: 1,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
  },
  resetButton: {
    backgroundColor: colors.surfaceVariant,
    borderWidth: 1,
    borderColor: colors.border,
  },
  applyButton: {
    backgroundColor: colors.primary,
  },
  resetButtonText: {
    ...typography.body,
    fontWeight: '600',
    color: colors.text,
  },
  applyButtonText: {
    ...typography.body,
    fontWeight: '600',
    color: colors.surface,
  },
});

