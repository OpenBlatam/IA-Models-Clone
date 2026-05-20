import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams } from 'expo-router';
import { ManualGeneratorForm } from '@/components/generate/manual-generator-form';
import { useApp } from '@/lib/context/app-context';

export default function GenerateScreen() {
  const params = useLocalSearchParams();
  const mode = (params.mode as string) || 'text';
  const category = (params.category as string) || undefined;
  const { state } = useApp();

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: state.colors.background }]} edges={['top']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          <ManualGeneratorForm initialMode={mode} initialCategory={category} />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  keyboardView: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
});




