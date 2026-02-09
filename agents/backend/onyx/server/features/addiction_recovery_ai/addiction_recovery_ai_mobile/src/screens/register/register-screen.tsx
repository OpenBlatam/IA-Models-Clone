import React from 'react';
import {
  View,
  Text,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Input, Button } from '@/components';
import { useRegister } from '@/hooks/api';
import { useAuthStore } from '@/store/auth-store';
import { useColors } from '@/theme/colors';
import { useRegisterStyles } from './register-screen.styles';
import { useRegisterForm } from './use-register-form';

export function RegisterScreen(): JSX.Element {
  const colors = useColors();
  const { register, error, isLoading } = useAuthStore();
  const registerMutation = useRegister();
  const styles = useRegisterStyles(colors);
  const { formState, handlers } = useRegisterForm(register, error);

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
            <Text style={styles.title} accessibilityRole="header">
              Crear Cuenta
            </Text>
            <Text style={styles.subtitle} accessibilityRole="text">
              Regístrate para comenzar
            </Text>

            <Input
              label="ID de Usuario *"
              value={formState.userId}
              onChangeText={handlers.setUserId}
              placeholder="Ingresa un ID único"
              autoCapitalize="none"
              error={formState.errors.userId}
              style={styles.input}
            />

            <Input
              label="Nombre (Opcional)"
              value={formState.name}
              onChangeText={handlers.setName}
              placeholder="Tu nombre"
              style={styles.input}
            />

            <Input
              label="Email (Opcional)"
              value={formState.email}
              onChangeText={handlers.setEmail}
              placeholder="tu@email.com"
              keyboardType="email-address"
              autoCapitalize="none"
              style={styles.input}
            />

            <Input
              label="Contraseña (Opcional)"
              value={formState.password}
              onChangeText={handlers.setPassword}
              placeholder="Mínimo 8 caracteres"
              secureTextEntry
              error={formState.errors.password}
              style={styles.input}
            />

            {error && (
              <Text style={styles.errorText} accessibilityRole="alert">
                {error}
              </Text>
            )}

            <Button
              title="Registrarse"
              onPress={handlers.handleRegister}
              loading={isLoading || registerMutation.isPending}
              disabled={!formState.isValid}
              style={styles.button}
            />
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

