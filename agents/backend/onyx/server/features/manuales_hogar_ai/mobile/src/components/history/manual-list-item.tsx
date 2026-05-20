/**
 * Manual List Item
 * ================
 * List item component for manual history
 */

import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { format } from 'date-fns';
import { es } from 'date-fns/locale/es';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import type { Manual } from '@/types/api';
import { CATEGORIES } from '@/constants/categories';

interface ManualListItemProps {
  manual: Manual;
}

export function ManualListItem({ manual }: ManualListItemProps) {
  const router = useRouter();
  const { state } = useApp();
  const colors = state.colors;

  const category = CATEGORIES[manual.category] || CATEGORIES.general;
  const date = new Date(manual.created_at);
  const formattedDate = format(date, "d 'de' MMM, yyyy", { locale: es });

  return (
    <TouchableOpacity
      style={[styles.container, { backgroundColor: colors.card }]}
      onPress={() => router.push(`/manual/${manual.id}` as any)}
      activeOpacity={0.7}
    >
      <View style={styles.header}>
        <View style={[styles.categoryBadge, { backgroundColor: `${category.color}20` }]}>
          <Ionicons name={category.icon as any} size={16} color={category.color} />
          <Text style={[styles.categoryText, { color: category.color }]}>
            {category.displayName}
          </Text>
        </View>
        <Text style={[styles.date, { color: colors.textSecondary }]}>{formattedDate}</Text>
      </View>
      <Text style={[styles.preview, { color: colors.text }]} numberOfLines={3}>
        {manual.manual?.substring(0, 150) || 'Sin contenido'}...
      </Text>
      {manual.tokens_used && (
        <View style={styles.footer}>
          <Text style={[styles.tokens, { color: colors.textSecondary }]}>
            {manual.tokens_used.toLocaleString()} tokens
          </Text>
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  categoryBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  categoryText: {
    fontSize: 12,
    fontWeight: '500',
  },
  date: {
    fontSize: 12,
  },
  preview: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 8,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  tokens: {
    fontSize: 12,
  },
});

