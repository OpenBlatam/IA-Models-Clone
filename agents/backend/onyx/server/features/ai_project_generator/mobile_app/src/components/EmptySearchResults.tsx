import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';

interface EmptySearchResultsProps {
  searchQuery: string;
  onClearSearch?: () => void;
  message?: string;
  icon?: React.ReactNode;
}

export const EmptySearchResults: React.FC<EmptySearchResultsProps> = ({
  searchQuery,
  onClearSearch,
  message,
  icon,
}) => {
  const { theme } = useTheme();

  return (
    <View style={styles.container}>
      {icon && <View style={styles.iconContainer}>{icon}</View>}
      <Text style={[styles.title, { color: theme.text }]}>
        {message || 'No se encontraron resultados'}
      </Text>
      <Text style={[styles.subtitle, { color: theme.textSecondary }]}>
        No hay resultados para "{searchQuery}"
      </Text>
      {onClearSearch && (
        <TouchableOpacity
          style={[styles.button, { backgroundColor: theme.primary }]}
          onPress={onClearSearch}
          activeOpacity={0.7}
        >
          <Text style={[styles.buttonText, { color: theme.surface }]}>
            Limpiar búsqueda
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  iconContainer: {
    marginBottom: spacing.lg,
  },
  title: {
    ...typography.h3,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  subtitle: {
    ...typography.body,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  button: {
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    borderRadius: 8,
  },
  buttonText: {
    ...typography.body,
    fontWeight: '600',
  },
});

