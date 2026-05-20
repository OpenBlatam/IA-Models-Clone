/**
 * Custom hook for online/offline status detection.
 * Provides real-time network status monitoring.
 */

import { useState, useEffect } from 'react';

/**
 * Online status interface.
 */
export interface OnlineStatus {
  isOnline: boolean;
  wasOffline: boolean;
}

/**
 * Custom hook for tracking online/offline status.
 * Monitors network connectivity and provides status updates.
 *
 * @returns Online status information
 */
export function useOnlineStatus(): OnlineStatus {
  const [isOnline, setIsOnline] = useState<boolean>(() => {
    if (typeof window === 'undefined') {
      return true; // Assume online on server
    }
    return navigator.onLine;
  });

  const [wasOffline, setWasOffline] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const handleOnline = () => {
      setIsOnline(true);
      setWasOffline(true);
    };

    const handleOffline = () => {
      setIsOnline(false);
      setWasOffline(false);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Check initial status
    setIsOnline(navigator.onLine);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return { isOnline, wasOffline };
}

