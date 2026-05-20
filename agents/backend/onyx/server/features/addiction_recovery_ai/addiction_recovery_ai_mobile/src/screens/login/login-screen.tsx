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
import { useLogin } from '@/hooks/api';
import { useAuthStore } from '@/store/auth-store';
import { useColors } from '@/theme/colors';
import { loginSchema, type LoginFormData } from '@/utils/validation';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useLoginStyles } from './login-screen.styles';

export function LoginScreen(): JSX.Element {
  const colors = useColors();
  const { login, error, isLoading } = useAuthStore();
  const loginMutation = useLogin();
  const styles = useLoginStyles(colors);

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

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          <View style={styles.content}>
            <Text
              style={styles.title}
              accessibilityRole="header"
            >
              Bienvenido
            </Text>
            <Text
              style={styles.subtitle}
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
                style={styles.errorText}
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
              <Text style={styles.footerText}>
                ¿No tienes cuenta?{' '}
              </Text>
              <TouchableOpacity
                accessibilityRole="button"
                accessibilityLabel="Ir a registro"
              >
                <Text style={styles.linkText}>
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

