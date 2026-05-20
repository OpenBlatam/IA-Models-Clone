/**
 * Manual Detail Content
 * =====================
 * Component for displaying manual content
 */

import { View, Text, StyleSheet } from 'react-native';
import { useApp } from '@/lib/context/app-context';
import type { Manual } from '@/types/api';
import { CATEGORIES } from '@/constants/categories';

interface ManualDetailContentProps {
  manual: Manual;
}

export function ManualDetailContent({ manual }: ManualDetailContentProps) {
  const { state } = useApp();
  const colors = state.colors;

  const category = CATEGORIES[manual.category] || CATEGORIES.general;

  return (
    <View style={styles.container}>
      <View style={[styles.categoryBadge, { backgroundColor: `${category.color}20` }]}>
        <Text style={[styles.categoryText, { color: category.color }]}>
          {category.displayName}
        </Text>
      </View>

      <Text style={[styles.content, { color: colors.text }]}>{manual.manual}</Text>

      {manual.tokens_used && (
        <View style={[styles.footer, { borderTopColor: colors.border }]}>
          <Text style={[styles.footerText, { color: colors.textSecondary }]}>
            Tokens usados: {manual.tokens_used.toLocaleString()}
          </Text>
          {manual.model_used && (
            <Text style={[styles.footerText, { color: colors.textSecondary }]}>
              Modelo: {manual.model_used}
            </Text>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: 16,
  },
  categoryBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  categoryText: {
    fontSize: 14,
    fontWeight: '600',
  },
  content: {
    fontSize: 16,
    lineHeight: 24,
  },
  footer: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    gap: 4,
  },
  footerText: {
    fontSize: 12,
  },
});




