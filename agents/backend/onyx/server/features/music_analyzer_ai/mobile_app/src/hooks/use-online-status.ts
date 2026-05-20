import { useState, useEffect } from 'react';
import { useNetworkStatus } from './use-network-status';

/**
 * Hook to check if device is online
 * Provides online/offline state with callbacks
 */
export function useOnlineStatus() {
  const { isConnected, isInternetReachable } = useNetworkStatus();
  const [wasOffline, setWasOffline] = useState(false);

  const isOnline = isConnected && (isInternetReachable ?? true);

  useEffect(() => {
    if (!isOnline && !wasOffline) {
      setWasOffline(true);
    } else if (isOnline && wasOffline) {
      setWasOffline(false);
    }
  }, [isOnline, wasOffline]);

  return {
    isOnline,
    isOffline: !isOnline,
    wasOffline,
    justCameOnline: isOnline && wasOffline,
    justWentOffline: !isOnline && !wasOffline,
  };
}

