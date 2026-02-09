import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Button } from '@/components/ui/button';

export default function CreateScreen() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.content}>
        <Text style={styles.title}>Create Video</Text>
        <Text style={styles.subtitle}>Choose how you want to create your video</Text>

        <Button
          title="Generate from Script"
          onPress={() => router.push('/video-generation')}
          variant="primary"
          size="large"
          style={styles.button}
        />

        <Button
          title="Use Template"
          onPress={() => router.push('/(tabs)/templates')}
          variant="secondary"
          size="large"
          style={styles.button}
        />

        <Button
          title="Batch Generation"
          onPress={() => router.push('/batch-generation')}
          variant="outline"
          size="large"
          style={styles.button}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  content: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#000000',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666666',
    marginBottom: 32,
    textAlign: 'center',
  },
  button: {
    marginBottom: 12,
  },
});


