import { useEffect, useState } from 'react';
import NetInfo from '@react-native-community/netinfo';

export function useNetworkStatus() {
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [connectionType, setConnectionType] = useState<string | null>(null);
  const [isInternetReachable, setIsInternetReachable] = useState<boolean | null>(null);

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      setIsConnected(state.isConnected);
      setConnectionType(state.type);
      setIsInternetReachable(state.isInternetReachable);
    });

    // Get initial state
    NetInfo.fetch().then((state) => {
      setIsConnected(state.isConnected);
      setConnectionType(state.type);
      setIsInternetReachable(state.isInternetReachable);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  return {
    isConnected,
    connectionType,
    isInternetReachable,
    isOffline: isConnected === false,
  };
}

export function useNetworkRequest<T>(
  requestFn: () => Promise<T>,
  options?: {
    retryOnNetworkError?: boolean;
    maxRetries?: number;
    retryDelay?: number;
  }
) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const { isConnected } = useNetworkStatus();

  const execute = async (): Promise<T | null> => {
    if (!isConnected) {
      setError(new Error('No internet connection'));
      return null;
    }

    setIsLoading(true);
    setError(null);

    let retries = 0;
    const maxRetries = options?.maxRetries || 3;
    const retryDelay = options?.retryDelay || 1000;

    while (retries <= maxRetries) {
      try {
        const result = await requestFn();
        setIsLoading(false);
        return result;
      } catch (err) {
        if (retries < maxRetries && options?.retryOnNetworkError) {
          retries++;
          await new Promise((resolve) => setTimeout(resolve, retryDelay * retries));
          continue;
        }
        setError(err as Error);
        setIsLoading(false);
        return null;
      }
    }

    setIsLoading(false);
    return null;
  };

  return { execute, isLoading, error };
}


