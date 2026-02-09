'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiCloud, FiCloudOff, FiRefreshCw } from 'react-icons/fi';

export default function SyncStatus() {
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    setIsOnline(navigator.onLine);

    const handleOnline = () => {
      setIsOnline(true);
      syncData();
    };

    const handleOffline = () => {
      setIsOnline(false);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Initial sync
    syncData();

    // Auto-sync every 30 seconds
    const interval = setInterval(syncData, 30000);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      clearInterval(interval);
    };
  }, []);

  const syncData = async () => {
    if (!navigator.onLine) return;

    setIsSyncing(true);
    try {
      // Simulate sync (in production, sync with backend)
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setLastSync(new Date());
    } catch (error) {
      console.error('Sync error:', error);
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="fixed bottom-4 left-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3 flex items-center gap-2 z-50"
      >
        {isOnline ? (
          <>
            {isSyncing ? (
              <FiRefreshCw size={16} className="text-primary-600 animate-spin" />
            ) : (
              <FiCloud size={16} className="text-green-500" />
            )}
            <div className="text-xs">
              <div className="text-gray-700 dark:text-gray-300 font-medium">
                {isSyncing ? 'Sincronizando...' : 'Sincronizado'}
              </div>
              {lastSync && (
                <div className="text-gray-500 dark:text-gray-400">
                  {lastSync.toLocaleTimeString()}
                </div>
              )}
            </div>
          </>
        ) : (
          <>
            <FiCloudOff size={16} className="text-red-500" />
            <div className="text-xs text-gray-700 dark:text-gray-300">
              Sin conexión
            </div>
          </>
        )}
      </motion.div>
    </AnimatePresence>
  );
}


