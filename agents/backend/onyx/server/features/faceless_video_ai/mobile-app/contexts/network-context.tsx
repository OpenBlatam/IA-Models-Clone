import React, { createContext, useContext, useEffect, useState } from 'react';
import { useNetworkStatus } from '@/hooks/use-network';
import { Alert } from 'react-native';

interface NetworkContextValue {
  isConnected: boolean | null;
  isOffline: boolean;
  connectionType: string | null;
  showOfflineBanner: boolean;
  setShowOfflineBanner: (show: boolean) => void;
}

const NetworkContext = createContext<NetworkContextValue | undefined>(undefined);

export function NetworkProvider({ children }: { children: React.ReactNode }) {
  const { isConnected, isOffline, connectionType } = useNetworkStatus();
  const [showOfflineBanner, setShowOfflineBanner] = useState(false);

  useEffect(() => {
    if (isOffline) {
      setShowOfflineBanner(true);
    } else {
      setShowOfflineBanner(false);
    }
  }, [isOffline]);

  const value: NetworkContextValue = {
    isConnected,
    isOffline,
    connectionType,
    showOfflineBanner,
    setShowOfflineBanner,
  };

  return (
    <NetworkContext.Provider value={value}>{children}</NetworkContext.Provider>
  );
}

export function useNetwork() {
  const context = useContext(NetworkContext);
  if (context === undefined) {
    throw new Error('useNetwork must be used within a NetworkProvider');
  }
  return context;
}


