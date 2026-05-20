import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Modal,
} from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { useDebouncedCallback } from '../hooks/useOptimizedCallback';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';
import { Project } from '../types';

interface QuickSearchProps {
  visible: boolean;
  onClose: () => void;
  onSelect: (project: Project) => void;
  projects: Project[];
  placeholder?: string;
}

export const QuickSearch: React.FC<QuickSearchProps> = ({
  visible,
  onClose,
  onSelect,
  projects,
  placeholder = 'Buscar proyecto...',
}) => {
  const { theme } = useTheme();
  const [query, setQuery] = useState('');
  const [filteredProjects, setFilteredProjects] = useState<Project[]>([]);

  const handleSearch = useDebouncedCallback(
    (searchQuery: string) => {
      if (!searchQuery.trim()) {
        setFilteredProjects([]);
        return;
      }

      const lowerQuery = searchQuery.toLowerCase();
      const filtered = projects.filter(
        (p) =>
          p.project_name.toLowerCase().includes(lowerQuery) ||
          p.description.toLowerCase().includes(lowerQuery) ||
          p.author.toLowerCase().includes(lowerQuery) ||
          p.project_id.toLowerCase().includes(lowerQuery)
      );
      setFilteredProjects(filtered);
      hapticFeedback.selection();
    },
    200,
    [projects]
  );

  const handleQueryChange = (text: string) => {
    setQuery(text);
    handleSearch(text);
  };

  const handleSelect = (project: Project) => {
    hapticFeedback.success();
    onSelect(project);
    setQuery('');
    setFilteredProjects([]);
    onClose();
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <TouchableOpacity
        style={styles.overlay}
        activeOpacity={1}
        onPress={onClose}
      >
        <View
          style={[styles.container, { backgroundColor: theme.surface }]}
          onStartShouldSetResponder={() => true}
        >
          <View style={[styles.header, { borderBottomColor: theme.border }]}>
            <TextInput
              style={[
                styles.input,
                {
                  backgroundColor: theme.surfaceVariant,
                  color: theme.text,
                  borderColor: theme.border,
                },
              ]}
              placeholder={placeholder}
              placeholderTextColor={theme.textTertiary}
              value={query}
              onChangeText={handleQueryChange}
              autoFocus
              returnKeyType="search"
            />
            <TouchableOpacity onPress={onClose} style={styles.closeButton}>
              <Text style={[styles.closeText, { color: theme.text }]}>✕</Text>
            </TouchableOpacity>
          </View>

          {filteredProjects.length > 0 && (
            <FlatList
              data={filteredProjects}
              keyExtractor={(item) => item.project_id}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={[styles.resultItem, { borderBottomColor: theme.border }]}
                  onPress={() => handleSelect(item)}
                >
                  <Text style={[styles.resultTitle, { color: theme.text }]} numberOfLines={1}>
                    {item.project_name}
                  </Text>
                  <Text style={[styles.resultSubtitle, { color: theme.textSecondary }]} numberOfLines={1}>
                    {item.author} • {item.status}
                  </Text>
                </TouchableOpacity>
              )}
              style={styles.resultsList}
              keyboardShouldPersistTaps="handled"
            />
          )}

          {query.trim() && filteredProjects.length === 0 && (
            <View style={styles.emptyContainer}>
              <Text style={[styles.emptyText, { color: theme.textSecondary }]}>
                No se encontraron resultados
              </Text>
            </View>
          )}
        </View>
      </TouchableOpacity>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-start',
    paddingTop: 100,
  },
  container: {
    marginHorizontal: spacing.lg,
    borderRadius: borderRadius.lg,
    maxHeight: 500,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  header: {
    flexDirection: 'row',
    padding: spacing.md,
    borderBottomWidth: 1,
    alignItems: 'center',
    gap: spacing.sm,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    ...typography.body,
  },
  closeButton: {
    padding: spacing.sm,
  },
  closeText: {
    fontSize: 20,
    fontWeight: '600',
  },
  resultsList: {
    maxHeight: 400,
  },
  resultItem: {
    padding: spacing.md,
    borderBottomWidth: 1,
  },
  resultTitle: {
    ...typography.body,
    fontWeight: '600',
    marginBottom: spacing.xs,
  },
  resultSubtitle: {
    ...typography.caption,
  },
  emptyContainer: {
    padding: spacing.xl,
    alignItems: 'center',
  },
  emptyText: {
    ...typography.body,
  },
});

