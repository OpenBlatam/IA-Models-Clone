import { useState } from 'react';
import * as LocalAuthentication from 'expo-local-authentication';

interface BiometricsState {
  isAvailable: boolean;
  isEnabled: boolean;
  type: string | null;
}

export const useBiometrics = () => {
  const [state, setState] = useState<BiometricsState>({
    isAvailable: false,
    isEnabled: false,
    type: null,
  });

  const checkAvailability = async () => {
    try {
      const compatible = await LocalAuthentication.hasHardwareAsync();
      const enrolled = await LocalAuthentication.isEnrolledAsync();
      const types = await LocalAuthentication.supportedAuthenticationTypesAsync();

      setState({
        isAvailable: compatible && enrolled,
        isEnabled: compatible && enrolled,
        type: types.length > 0 ? 'biometric' : null,
      });

      return compatible && enrolled;
    } catch (error) {
      console.error('Error checking biometrics:', error);
      return false;
    }
  };

  const authenticate = async (reason?: string): Promise<boolean> => {
    try {
      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: reason || 'Autenticación requerida',
        cancelLabel: 'Cancelar',
        disableDeviceFallback: false,
      });

      return result.success;
    } catch (error) {
      console.error('Error authenticating:', error);
      return false;
    }
  };

  return {
    ...state,
    checkAvailability,
    authenticate,
  };
};

