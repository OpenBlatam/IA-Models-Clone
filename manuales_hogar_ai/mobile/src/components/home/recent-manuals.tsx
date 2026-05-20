/**
 * Recent Manuals
 * ==============
 * List of recently generated manuals
 */

import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { format } from 'date-fns';
import { es } from 'date-fns/locale/es';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import type { Manual } from '@/types/api';
import { CATEGORIES } from '@/constants/categories';

interface RecentManualsProps {
  manuals: { manuals: Manual[] };
}

export function RecentManuals({ manuals }: RecentManualsProps) {
  const router = useRouter();
  const { state } = useApp();
  const colors = state.colors;

  if (!manuals?.manuals || manuals.manuals.length === 0) {
    return (
      <View style={styles.container}>
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Manuales Recientes</Text>
        <View style={[styles.emptyState, { backgroundColor: colors.card }]}>
          <Ionicons name="document-text-outline" size={48} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
            No hay manuales recientes
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Manuales Recientes</Text>
        <TouchableOpacity onPress={() => router.push('/(tabs)/history' as any)}>
          <Text style={[styles.seeAll, { color: colors.tint }]}>Ver todos</Text>
        </TouchableOpacity>
      </View>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {manuals.manuals.map((manual) => {
          const category = CATEGORIES[manual.category] || CATEGORIES.general;
          const date = new Date(manual.created_at);
          const formattedDate = format(date, "d 'de' MMM", { locale: es });

          return (
            <TouchableOpacity
              key={manual.id}
              style={[styles.manualCard, { backgroundColor: colors.card }]}
              onPress={() => router.push(`/manual/${manual.id}` as any)}
              activeOpacity={0.7}
            >
              <View style={[styles.categoryBadge, { backgroundColor: `${category.color}20` }]}>
                <Ionicons name={category.icon as any} size={16} color={category.color} />
                <Text style={[styles.categoryText, { color: category.color }]}>
                  {category.displayName}
                </Text>
              </View>
              <Text style={[styles.manualPreview, { color: colors.text }]} numberOfLines={3}>
                {manual.manual?.substring(0, 100) || 'Sin contenido'}...
              </Text>
              <Text style={[styles.dateText, { color: colors.textSecondary }]}>
                {formattedDate}
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginTop: 24,
    marginBottom: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
  },
  seeAll: {
    fontSize: 14,
    fontWeight: '500',
  },
  scrollContent: {
    paddingHorizontal: 20,
    gap: 12,
  },
  manualCard: {
    width: 280,
    padding: 16,
    borderRadius: 12,
    marginRight: 12,
  },
  categoryBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    marginBottom: 12,
    gap: 4,
  },
  categoryText: {
    fontSize: 12,
    fontWeight: '500',
  },
  manualPreview: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 8,
  },
  dateText: {
    fontSize: 12,
  },
  emptyState: {
    marginHorizontal: 20,
    padding: 32,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyText: {
    fontSize: 14,
    marginTop: 12,
    textAlign: 'center',
  },
});

