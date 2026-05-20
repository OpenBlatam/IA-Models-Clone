import { useState, useCallback } from 'react';
import { Alert } from 'react-native';
import type { RegisterRequest } from '@/types';

interface RegisterFormState {
  userId: string;
  email: string;
  name: string;
  password: string;
  errors: {
    userId?: string;
    password?: string;
  };
  isValid: boolean;
}

interface RegisterFormHandlers {
  setUserId: (value: string) => void;
  setEmail: (value: string) => void;
  setName: (value: string) => void;
  setPassword: (value: string) => void;
  handleRegister: () => Promise<void>;
}

export function useRegisterForm(
  register: (data: RegisterRequest) => Promise<void>,
  error: string | null
): { formState: RegisterFormState; handlers: RegisterFormHandlers } {
  const [userId, setUserId] = useState('');
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<{ userId?: string; password?: string }>({});

  const validate = useCallback((): boolean => {
    const newErrors: { userId?: string; password?: string } = {};

    if (!userId.trim()) {
      newErrors.userId = 'Por favor ingresa un ID de usuario';
    }

    if (password && password.length < 8) {
      newErrors.password = 'La contraseña debe tener al menos 8 caracteres';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0 && userId.trim().length > 0;
  }, [userId, password]);

  const handleRegister = useCallback(async () => {
    if (!validate()) {
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
  }, [userId, email, name, password, register, error, validate]);

  return {
    formState: {
      userId,
      email,
      name,
      password,
      errors,
      isValid: validate(),
    },
    handlers: {
      setUserId,
      setEmail,
      setName,
      setPassword,
      handleRegister,
    },
  };
}

