import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert,
} from 'react-native';
import { Input, Button } from '@/components';
import { useRegister } from '@/hooks/useApi';
import { useAuthStore } from '@/store/auth-store';

export const RegisterScreen: React.FC = () => {
  const [userId, setUserId] = useState('');
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const { register, error, isLoading } = useAuthStore();
  const registerMutation = useRegister();

  const handleRegister = async () => {
    if (!userId.trim()) {
      Alert.alert('Error', 'Por favor ingresa un ID de usuario');
      return;
    }

    if (password && password.length < 8) {
      Alert.alert('Error', 'La contraseña debe tener al menos 8 caracteres');
      return;
    }

    try {
      await register({
        user_id: userId,
        email: email || undefined,
        name: name || undefined,
        password: password || undefined,
      });
    } catch (err) {
      Alert.alert('Error', error || 'Error al registrarse');
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.content}>
          <Text style={styles.title}>Crear Cuenta</Text>
          <Text style={styles.subtitle}>Regístrate para comenzar</Text>

          <Input
            label="ID de Usuario *"
            value={userId}
            onChangeText={setUserId}
            placeholder="Ingresa un ID único"
            autoCapitalize="none"
            style={styles.input}
          />

          <Input
            label="Nombre (Opcional)"
            value={name}
            onChangeText={setName}
            placeholder="Tu nombre"
            style={styles.input}
          />

          <Input
            label="Email (Opcional)"
            value={email}
            onChangeText={setEmail}
            placeholder="tu@email.com"
            keyboardType="email-address"
            autoCapitalize="none"
            style={styles.input}
          />

          <Input
            label="Contraseña (Opcional)"
            value={password}
            onChangeText={setPassword}
            placeholder="Mínimo 8 caracteres"
            secureTextEntry
            style={styles.input}
          />

          {error && <Text style={styles.errorText}>{error}</Text>}

          <Button
            title="Registrarse"
            onPress={handleRegister}
            loading={isLoading || registerMutation.isPending}
            style={styles.button}
          />
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 24,
  },
  content: {
    backgroundColor: '#FFF',
    borderRadius: 16,
    padding: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 32,
    textAlign: 'center',
  },
  input: {
    marginBottom: 16,
  },
  button: {
    marginTop: 8,
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 14,
    marginBottom: 16,
    textAlign: 'center',
  },
});

