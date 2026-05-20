import { useState, useEffect } from 'react';
import NetInfo from '@react-native-community/netinfo';

export interface NetworkState {
  isConnected: boolean | null;
  type: string | null;
  isInternetReachable: boolean | null;
  details: any;
}

export function useNetworkStatus() {
  const [networkState, setNetworkState] = useState<NetworkState>({
    isConnected: null,
    type: null,
    isInternetReachable: null,
    details: null,
  });

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      setNetworkState({
        isConnected: state.isConnected,
        type: state.type,
        isInternetReachable: state.isInternetReachable,
        details: state.details,
      });
    });

    // Get initial state
    NetInfo.fetch().then((state) => {
      setNetworkState({
        isConnected: state.isConnected,
        type: state.type,
        isInternetReachable: state.isInternetReachable,
        details: state.details,
      });
    });

    return () => {
      unsubscribe();
    };
  }, []);

  return networkState;
}


