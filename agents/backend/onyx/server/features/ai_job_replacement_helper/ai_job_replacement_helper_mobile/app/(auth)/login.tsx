import React, { useCallback, useMemo } from 'react';
import { View, Text, StyleSheet, KeyboardAvoidingView, Platform, ScrollView, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '@/store/authStore';
import { useForm } from '@/hooks/useForm';
import { loginSchema, type LoginFormData } from '@/utils/validation';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card } from '@/components/ui/Card';
import { useTheme } from '@/theme/theme';

export default function LoginScreen() {
  const { login, isLoading, error, clearError } = useAuthStore();
  const router = useRouter();
  const theme = useTheme();

  const form = useForm<LoginFormData>({
    initialValues: {
      email: '',
      password: '',
    },
    validationSchema: loginSchema,
    onSubmit: async (values) => {
      clearError();
      const success = await login(values.email, values.password);

      if (success) {
        router.replace('/(tabs)/dashboard');
      } else {
        Alert.alert('Login Failed', error || 'Invalid credentials');
      }
    },
  });

  const handleRegisterPress = useCallback(() => {
    router.push('/(auth)/register');
  }, [router]);

  const containerStyle = useMemo(
    () => [styles.container, { backgroundColor: theme.colors.background }],
    [theme.colors.background]
  );

  return (
    <SafeAreaView style={containerStyle} edges={['top', 'bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.content}>
            <View style={styles.header}>
              <Text style={[styles.title, { color: theme.colors.text }]}>
                AI Job Replacement Helper
              </Text>
              <Text style={[styles.subtitle, { color: theme.colors.textSecondary }]}>
                Sign in to continue
              </Text>
            </View>

            <Card style={styles.card}>
              <Input
                label="Email"
                value={form.values.email}
                onChangeText={form.handleChange('email')}
                onBlur={form.handleBlur('email')}
                error={form.touched.email ? form.errors.email : undefined}
                keyboardType="email-address"
                autoCapitalize="none"
                autoComplete="email"
                autoCorrect={false}
                leftIcon={<Ionicons name="mail-outline" size={20} color={theme.colors.textSecondary} />}
                accessibilityLabel="Email address"
                accessibilityHint="Enter your email address"
              />

              <Input
                label="Password"
                value={form.values.password}
                onChangeText={form.handleChange('password')}
                onBlur={form.handleBlur('password')}
                error={form.touched.password ? form.errors.password : undefined}
                secureTextEntry
                autoCapitalize="none"
                autoComplete="password"
                autoCorrect={false}
                leftIcon={<Ionicons name="lock-closed-outline" size={20} color={theme.colors.textSecondary} />}
                accessibilityLabel="Password"
                accessibilityHint="Enter your password"
              />

              <Button
                title="Sign In"
                onPress={form.handleSubmit}
                loading={isLoading}
                disabled={isLoading || !form.isValid}
                fullWidth
                style={styles.button}
                accessibilityLabel="Sign in to your account"
                accessibilityHint="Tap to sign in with your email and password"
              />

              <Button
                title="Don't have an account? Sign Up"
                onPress={handleRegisterPress}
                variant="ghost"
                size="small"
                fullWidth
                style={styles.linkButton}
                accessibilityLabel="Navigate to registration screen"
              />
            </Card>
          </View>
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
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
  },
  content: {
    padding: 20,
  },
  header: {
    marginBottom: 32,
    alignItems: 'center',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
  },
  card: {
    padding: 20,
  },
  button: {
    marginTop: 8,
  },
  linkButton: {
    marginTop: 16,
  },
});

