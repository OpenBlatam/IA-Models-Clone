import { useState, useEffect } from 'react';
import { getNetworkStatus, isOnline, getConnectionType, getConnectionSpeed, isSlowConnection } from '@/lib/utils/network';

export function useNetwork() {
  const [status, setStatus] = useState(getNetworkStatus());

  useEffect(() => {
    const updateStatus = () => {
      setStatus(getNetworkStatus());
    };

    window.addEventListener('online', updateStatus);
    window.addEventListener('offline', updateStatus);

    // Listen for connection changes
    if ('connection' in navigator) {
      const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection;
      if (connection) {
        connection.addEventListener('change', updateStatus);
      }
    }

    return () => {
      window.removeEventListener('online', updateStatus);
      window.removeEventListener('offline', updateStatus);
      
      if ('connection' in navigator) {
        const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection;
        if (connection) {
          connection.removeEventListener('change', updateStatus);
        }
      }
    };
  }, []);

  return {
    ...status,
    isOnline: status.online,
    isOffline: !status.online,
    isSlow: status.slow,
  };
}



