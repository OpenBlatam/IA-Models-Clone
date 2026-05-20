import { useState, useCallback } from 'react';
import * as LocalAuthentication from 'expo-local-authentication';
import { Alert, Platform } from 'react-native';

export interface BiometricAuthResult {
  success: boolean;
  error?: string;
}

export function useBiometrics() {
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);
  const [biometricType, setBiometricType] = useState<LocalAuthentication.AuthenticationType[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const checkAvailability = useCallback(async () => {
    try {
      const compatible = await LocalAuthentication.hasHardwareAsync();
      if (!compatible) {
        setIsAvailable(false);
        return false;
      }

      const enrolled = await LocalAuthentication.isEnrolledAsync();
      if (!enrolled) {
        setIsAvailable(false);
        return false;
      }

      const types = await LocalAuthentication.supportedAuthenticationTypesAsync();
      setBiometricType(types);
      setIsAvailable(true);
      return true;
    } catch (error) {
      console.error('Error checking biometric availability:', error);
      setIsAvailable(false);
      return false;
    }
  }, []);

  const authenticate = useCallback(
    async (promptMessage: string = 'Authenticate to continue'): Promise<BiometricAuthResult> => {
      try {
        setIsLoading(true);

        const result = await LocalAuthentication.authenticateAsync({
          promptMessage,
          cancelLabel: 'Cancel',
          disableDeviceFallback: false,
        });

        if (result.success) {
          return { success: true };
        } else {
          return {
            success: false,
            error: result.error || 'Authentication failed',
          };
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Authentication error';
        return {
          success: false,
          error: errorMessage,
        };
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  return {
    isAvailable,
    biometricType,
    isLoading,
    checkAvailability,
    authenticate,
  };
}


