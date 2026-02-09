import React, { useState, useEffect } from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { useBiometrics } from '../hooks/useBiometrics';
import PermissionRequest from './PermissionRequest';

interface SecureViewProps {
  children: React.ReactNode;
  reason?: string;
  fallback?: React.ReactNode;
}

const SecureView: React.FC<SecureViewProps> = ({
  children,
  reason = 'Esta sección requiere autenticación',
  fallback,
}) => {
  const { isAvailable, authenticate, checkAvailability } = useBiometrics();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const init = async () => {
      const available = await checkAvailability();
      if (!available) {
        setIsAuthenticated(true); // Allow access if biometrics not available
      }
      setIsChecking(false);
    };
    init();
  }, []);

  const handleAuthenticate = async () => {
    const success = await authenticate(reason);
    setIsAuthenticated(success);
  };

  if (isChecking) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#6366f1" />
      </View>
    );
  }

  if (!isAvailable) {
    return <>{children}</>;
  }

  if (!isAuthenticated) {
    return (
      fallback || (
        <View style={styles.container}>
          <PermissionRequest
            title="Autenticación Requerida"
            message={reason}
            icon="lock-closed"
            onRequest={handleAuthenticate}
          />
        </View>
      )
    );
  }

  return <>{children}</>;
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
});

export default SecureView;

