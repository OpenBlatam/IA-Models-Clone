import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '@/contexts/theme-context';
import type { Template } from '@/types/api';

interface TemplateCardProps {
  template: Template;
  onPress?: () => void;
  selected?: boolean;
}

export function TemplateCard({ template, onPress, selected = false }: TemplateCardProps) {
  const { colors } = useTheme();

  return (
    <TouchableOpacity
      style={[
        styles.card,
        {
          backgroundColor: selected ? colors.primary + '20' : colors.card,
          borderColor: selected ? colors.primary : colors.border,
        },
      ]}
      onPress={onPress}
      activeOpacity={0.7}
    >
      <Text style={[styles.name, { color: colors.text }]}>{template.name}</Text>
      {template.description && (
        <Text style={[styles.description, { color: colors.textSecondary }]} numberOfLines={2}>
          {template.description}
        </Text>
      )}
      {selected && (
        <View style={[styles.selectedBadge, { backgroundColor: colors.primary }]}>
          <Text style={styles.selectedText}>Selected</Text>
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 2,
    position: 'relative',
  },
  name: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  description: {
    fontSize: 14,
    lineHeight: 20,
  },
  selectedBadge: {
    position: 'absolute',
    top: 8,
    right: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  selectedText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
  },
});

