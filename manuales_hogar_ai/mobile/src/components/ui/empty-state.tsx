/**
 * Empty State
 * ===========
 * Empty state component
 */

import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';

interface EmptyStateProps {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  message?: string;
}

export function EmptyState({ icon, title, message }: EmptyStateProps) {
  const { state } = useApp();
  const colors = state.colors;

  return (
    <View style={styles.container}>
      <Ionicons name={icon} size={64} color={colors.textSecondary} />
      <Text style={[styles.title, { color: colors.text }]}>{title}</Text>
      {message && <Text style={[styles.message, { color: colors.textSecondary }]}>{message}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    marginTop: 16,
    textAlign: 'center',
  },
  message: {
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },
});




