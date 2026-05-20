import { useState, useEffect } from 'react';
import { offlineManager } from '@/utils/offline-manager';

export function useOffline(): {
  isOnline: boolean;
  queueLength: number;
} {
  const [isOnline, setIsOnline] = useState(offlineManager.getIsOnline());
  const [queueLength, setQueueLength] = useState(offlineManager.getQueueLength());

  useEffect(() => {
    const unsubscribe = offlineManager.subscribe((online) => {
      setIsOnline(online);
      setQueueLength(offlineManager.getQueueLength());
    });

    return unsubscribe;
  }, []);

  return {
    isOnline,
    queueLength,
  };
}

