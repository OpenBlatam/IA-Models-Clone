import { useState, useEffect } from 'react';

export interface NetworkStatus {
  isConnected: boolean;
  isInternetReachable: boolean | null;
  type: string | null;
}

export const useNetworkStatus = () => {
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus>({
    isConnected: true,
    isInternetReachable: true,
    type: null,
  });

  useEffect(() => {
    let mounted = true;

    const checkNetwork = async () => {
      try {
        const NetInfo = await import('@react-native-community/netinfo');
        const unsubscribe = NetInfo.default.addEventListener((state) => {
          if (mounted) {
            setNetworkStatus({
              isConnected: state.isConnected ?? false,
              isInternetReachable: state.isInternetReachable,
              type: state.type,
            });
          }
        });

        return () => {
          mounted = false;
          unsubscribe();
        };
      } catch (error) {
        console.warn('NetInfo not available:', error);
        if (mounted) {
          setNetworkStatus({
            isConnected: true,
            isInternetReachable: true,
            type: null,
          });
        }
      }
    };

    checkNetwork();

    return () => {
      mounted = false;
    };
  }, []);

  return networkStatus;
};

