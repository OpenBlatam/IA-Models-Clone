/**
 * Quick Actions
 * =============
 * Quick action buttons for home screen
 */

import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';

interface QuickAction {
  id: string;
  title: string;
  icon: keyof typeof Ionicons.glyphMap;
  route: string;
  color: string;
}

const quickActions: QuickAction[] = [
  {
    id: 'text',
    title: 'Desde Texto',
    icon: 'text',
    route: '/(tabs)/generate?mode=text',
    color: '#007AFF',
  },
  {
    id: 'camera',
    title: 'Desde Foto',
    icon: 'camera',
    route: '/(tabs)/generate?mode=image',
    color: '#34C759',
  },
  {
    id: 'gallery',
    title: 'Desde Galería',
    icon: 'images',
    route: '/(tabs)/generate?mode=gallery',
    color: '#FF9500',
  },
];

export function QuickActions() {
  const router = useRouter();
  const { state } = useApp();
  const colors = state.colors;

  return (
    <View style={styles.container}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>Acciones Rápidas</Text>
      <View style={styles.actionsGrid}>
        {quickActions.map((action) => (
          <TouchableOpacity
            key={action.id}
            style={[styles.actionCard, { backgroundColor: colors.card }]}
            onPress={() => router.push(action.route as any)}
            activeOpacity={0.7}
          >
            <View style={[styles.iconContainer, { backgroundColor: `${action.color}20` }]}>
              <Ionicons name={action.icon} size={24} color={action.color} />
            </View>
            <Text style={[styles.actionTitle, { color: colors.text }]}>{action.title}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 20,
    marginTop: 24,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 12,
  },
  actionsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  actionCard: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    minHeight: 100,
    justifyContent: 'center',
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  actionTitle: {
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
});




