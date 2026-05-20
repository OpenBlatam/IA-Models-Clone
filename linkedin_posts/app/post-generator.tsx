import React from 'react';
import { View, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import LinkedInPostGenerator from '../components/ui/linkedin-post-generator';

export default function PostGeneratorPage() {
  return (
    <SafeAreaView style={styles.container}>
      <LinkedInPostGenerator />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
}); 