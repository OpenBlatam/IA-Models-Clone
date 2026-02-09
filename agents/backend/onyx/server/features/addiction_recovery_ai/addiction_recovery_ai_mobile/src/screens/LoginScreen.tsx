import React, { useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Input, Button } from '@/components';
import { useLogin } from '@/hooks/useApi';
import { useAuthStore } from '@/store/auth-store';
import { useColors } from '@/theme/colors';
import { loginSchema, type LoginFormData } from '@/utils/validation';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';

export function LoginScreen(): JSX.Element {
  const colors = useColors();
  const { login, error, isLoading } = useAuthStore();
  const loginMutation = useLogin();

  const {
    control,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    mode: 'onChange',
  });

  const onSubmit = useCallback(
    async (data: LoginFormData) => {
      try {
        await login({
          user_id: data.user_id,
          password: data.password || undefined,
        });
      } catch (err) {
        Alert.alert('Error', error || 'Error al iniciar sesión');
      }
    },
    [login, error]
  );

  const containerStyles = useCallback(
    () => [
      styles.container,
      { backgroundColor: colors.background },
    ],
    [colors.background]
  );

  const contentStyles = useCallback(
    () => [
      styles.content,
      {
        backgroundColor: colors.surface,
        shadowColor: colors.shadow,
      },
    ],
    [colors.surface, colors.shadow]
  );

  return (
    <SafeAreaView style={containerStyles()} edges={['top', 'bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          <View style={contentStyles()}>
            <Text
              style={[styles.title, { color: colors.text }]}
              accessibilityRole="header"
            >
              Bienvenido
            </Text>
            <Text
              style={[styles.subtitle, { color: colors.textSecondary }]}
              accessibilityRole="text"
            >
              Inicia sesión para continuar
            </Text>

            <Controller
              control={control}
              name="user_id"
              render={({ field: { onChange, onBlur, value } }) => (
                <Input
                  label="ID de Usuario"
                  value={value}
                  onChangeText={onChange}
                  onBlur={onBlur}
                  placeholder="Ingresa tu ID de usuario"
                  autoCapitalize="none"
                  error={errors.user_id?.message}
                  accessibilityLabel="Campo de ID de usuario"
                  accessibilityHint="Ingresa tu ID de usuario para iniciar sesión"
                />
              )}
            />

            <Controller
              control={control}
              name="password"
              render={({ field: { onChange, onBlur, value } }) => (
                <Input
                  label="Contraseña (Opcional)"
                  value={value || ''}
                  onChangeText={onChange}
                  onBlur={onBlur}
                  placeholder="Ingresa tu contraseña"
                  secureTextEntry
                  error={errors.password?.message}
                  accessibilityLabel="Campo de contraseña"
                  accessibilityHint="Ingresa tu contraseña si tienes una"
                />
              )}
            />

            {error && (
              <Text
                style={[styles.errorText, { color: colors.error }]}
                accessibilityRole="alert"
                accessibilityLiveRegion="polite"
              >
                {error}
              </Text>
            )}

            <Button
              title="Iniciar Sesión"
              onPress={handleSubmit(onSubmit)}
              loading={isLoading || loginMutation.isPending}
              disabled={!isValid}
              style={styles.button}
              accessibilityLabel="Botón para iniciar sesión"
              accessibilityHint="Presiona para iniciar sesión con tus credenciales"
            />

            <View style={styles.footer}>
              <Text style={[styles.footerText, { color: colors.textSecondary }]}>
                ¿No tienes cuenta?{' '}
              </Text>
              <TouchableOpacity
                accessibilityRole="button"
                accessibilityLabel="Ir a registro"
              >
                <Text style={[styles.linkText, { color: colors.primary }]}>
                  Regístrate
                </Text>
              </TouchableOpacity>
            </View>
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
    padding: 24,
  },
  content: {
    borderRadius: 16,
    padding: 24,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 32,
    textAlign: 'center',
  },
  button: {
    marginTop: 8,
    marginBottom: 16,
  },
  errorText: {
    fontSize: 14,
    marginBottom: 16,
    textAlign: 'center',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  footerText: {
    fontSize: 14,
  },
  linkText: {
    fontSize: 14,
    fontWeight: '600',
  },
});
