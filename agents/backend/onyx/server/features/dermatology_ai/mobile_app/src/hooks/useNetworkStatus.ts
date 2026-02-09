import { useState, useEffect } from 'react';
// Note: Install @react-native-community/netinfo for full functionality
// For now, using a simplified version

interface NetworkState {
  isConnected: boolean | null;
  type: string | null;
  isInternetReachable: boolean | null;
}

export const useNetworkStatus = () => {
  const [networkState, setNetworkState] = useState<NetworkState>({
    isConnected: true, // Default to true, will be updated when NetInfo is installed
    type: null,
    isInternetReachable: true,
  });

  useEffect(() => {
    // Try to use NetInfo if available
    try {
      // Dynamic import to avoid errors if not installed
      const NetInfo = require('@react-native-community/netinfo');
      
      const unsubscribe = NetInfo.addEventListener((state: any) => {
        setNetworkState({
          isConnected: state.isConnected,
          type: state.type,
          isInternetReachable: state.isInternetReachable,
        });
      });

      NetInfo.fetch().then((state: any) => {
        setNetworkState({
          isConnected: state.isConnected,
          type: state.type,
          isInternetReachable: state.isInternetReachable,
        });
      });

      return () => {
        unsubscribe();
      };
    } catch (error) {
      // NetInfo not installed, use default values
      console.log('NetInfo not available, using default network state');
    }
  }, []);

  return networkState;
};

