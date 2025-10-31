import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useSafeArea } from '../hooks/use-safe-area';
import LinkedInPostGenerator from '../ui/linkedin-post-generator';

export function LinkedInPostsScreen() {
  const { isTablet, getSafeAreaStyle } = useSafeArea();

  return (
    <SafeAreaView style={[styles.container, getSafeAreaStyle()]}>
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={[styles.title, isTablet && styles.titleTablet]}>
            LinkedIn Posts
          </Text>
          <Text style={[styles.subtitle, isTablet && styles.subtitleTablet]}>
            Create engaging posts with AI-powered optimization
          </Text>
        </View>

        {/* Main Content */}
        <View style={styles.content}>
          <LinkedInPostGenerator />
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
    flexGrow: 1,
    paddingHorizontal: 20,
  },
  header: {
    alignItems: 'center',
    paddingVertical: 20,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1a1a1a',
    textAlign: 'center',
    marginBottom: 8,
  },
  titleTablet: {
    fontSize: 36,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    lineHeight: 22,
  },
  subtitleTablet: {
    fontSize: 18,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
});

export default LinkedInPostsScreen; 