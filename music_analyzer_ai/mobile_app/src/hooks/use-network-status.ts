import { useState, useEffect } from 'react';
import NetInfo from '@react-native-community/netinfo';
import type { NetInfoState } from '@react-native-community/netinfo';

interface NetworkStatus {
  isConnected: boolean;
  isInternetReachable: boolean | null;
  type: string | null;
  details: NetInfoState['details'] | null;
}

/**
 * Hook to monitor network connectivity
 * Provides real-time network status updates
 */
export function useNetworkStatus(): NetworkStatus {
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus>({
    isConnected: true,
    isInternetReachable: null,
    type: null,
    details: null,
  });

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      setNetworkStatus({
        isConnected: state.isConnected ?? false,
        isInternetReachable: state.isInternetReachable,
        type: state.type,
        details: state.details,
      });
    });

    // Get initial state
    NetInfo.fetch().then((state) => {
      setNetworkStatus({
        isConnected: state.isConnected ?? false,
        isInternetReachable: state.isInternetReachable,
        type: state.type,
        details: state.details,
      });
    });

    return () => {
      unsubscribe();
    };
  }, []);

  return networkStatus;
}

/**
 * Hook to check if device is online
 */
export function useIsOnline(): boolean {
  const { isConnected, isInternetReachable } = useNetworkStatus();
  return isConnected && (isInternetReachable ?? true);
}

