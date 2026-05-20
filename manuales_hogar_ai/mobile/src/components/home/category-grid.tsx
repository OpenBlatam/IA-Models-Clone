/**
 * Category Grid
 * =============
 * Grid of category cards
 */

import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import { CATEGORY_LIST } from '@/constants/categories';

export function CategoryGrid({ categories }: { categories?: unknown[] }) {
  const router = useRouter();
  const { state } = useApp();
  const colors = state.colors;

  return (
    <View style={styles.container}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>Categorías</Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {CATEGORY_LIST.map((category) => (
          <TouchableOpacity
            key={category.id}
            style={[styles.categoryCard, { backgroundColor: colors.card }]}
            onPress={() => router.push(`/(tabs)/generate?category=${category.id}` as any)}
            activeOpacity={0.7}
          >
            <View style={[styles.iconContainer, { backgroundColor: `${category.color}20` }]}>
              <Ionicons name={category.icon as any} size={28} color={category.color} />
            </View>
            <Text style={[styles.categoryName, { color: colors.text }]}>{category.displayName}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginTop: 24,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 12,
    paddingHorizontal: 20,
  },
  scrollContent: {
    paddingHorizontal: 20,
    gap: 12,
  },
  categoryCard: {
    width: 120,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginRight: 12,
  },
  iconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  categoryName: {
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
});




