import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

export default function HomePage() {
  const router = useRouter();

  const navigationItems = [
    {
      title: 'Create Post',
      subtitle: 'Generate AI-optimized LinkedIn posts',
      icon: 'create-outline',
      route: '/post-generator',
      color: '#007AFF',
    },
    {
      title: 'Post History',
      subtitle: 'View and manage your generated posts',
      icon: 'time-outline',
      route: '/post-history',
      color: '#34C759',
    },
    {
      title: 'Analytics',
      subtitle: 'Track post performance and engagement',
      icon: 'analytics-outline',
      route: '/analytics',
      color: '#FF9500',
    },
    {
      title: 'Settings',
      subtitle: 'Configure app preferences and API settings',
      icon: 'settings-outline',
      route: '/settings',
      color: '#AF52DE',
    },
  ];

  function handleNavigation(route: string) {
    router.push(route as any);
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>LinkedIn Posts</Text>
          <Text style={styles.subtitle}>
            Create engaging posts with AI-powered optimization
          </Text>
        </View>

        {/* Navigation Grid */}
        <View style={styles.navigationGrid}>
          {navigationItems.map((item, index) => (
            <TouchableOpacity
              key={index}
              style={[styles.navigationCard, { borderLeftColor: item.color }]}
              onPress={() => handleNavigation(item.route)}
              activeOpacity={0.7}
            >
              <View style={styles.cardContent}>
                <View style={[styles.iconContainer, { backgroundColor: item.color }]}>
                  <Ionicons name={item.icon as any} size={24} color="#fff" />
                </View>
                <View style={styles.textContainer}>
                  <Text style={styles.cardTitle}>{item.title}</Text>
                  <Text style={styles.cardSubtitle}>{item.subtitle}</Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#ccc" />
              </View>
            </TouchableOpacity>
          ))}
        </View>

        {/* Quick Actions */}
        <View style={styles.quickActions}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
          <View style={styles.actionButtons}>
            <TouchableOpacity
              style={[styles.actionButton, { backgroundColor: '#007AFF' }]}
              onPress={() => handleNavigation('/post-generator')}
            >
              <Ionicons name="add" size={20} color="#fff" />
              <Text style={styles.actionButtonText}>New Post</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[styles.actionButton, { backgroundColor: '#34C759' }]}
              onPress={() => handleNavigation('/post-history')}
            >
              <Ionicons name="list" size={20} color="#fff" />
              <Text style={styles.actionButtonText}>View History</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    paddingVertical: 20,
    marginBottom: 30,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1a1a1a',
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    lineHeight: 22,
  },
  navigationGrid: {
    marginBottom: 30,
  },
  navigationCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    marginBottom: 16,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  cardContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  textContainer: {
    flex: 1,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1a1a1a',
    marginBottom: 4,
  },
  cardSubtitle: {
    fontSize: 14,
    color: '#666',
    lineHeight: 18,
  },
  quickActions: {
    marginTop: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1a1a1a',
    marginBottom: 16,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
}); 