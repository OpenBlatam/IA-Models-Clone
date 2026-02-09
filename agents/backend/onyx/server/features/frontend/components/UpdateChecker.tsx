'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiDownload, FiX, FiCheckCircle } from 'react-icons/fi';

export default function UpdateChecker() {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [isChecking, setIsChecking] = useState(false);

  useEffect(() => {
    // Check for updates periodically
    const checkForUpdates = async () => {
      setIsChecking(true);
      try {
        // In production, this would check against an API
        // For now, simulate update check
        const lastCheck = localStorage.getItem('bul_last_update_check');
        const now = Date.now();
        
        // Check every 24 hours
        if (!lastCheck || now - parseInt(lastCheck) > 24 * 60 * 60 * 1000) {
          // Simulate update available (in production, check version)
          const hasUpdate = Math.random() > 0.7; // 30% chance
          setUpdateAvailable(hasUpdate);
          localStorage.setItem('bul_last_update_check', now.toString());
        }
      } catch (error) {
        console.error('Update check error:', error);
      } finally {
        setIsChecking(false);
      }
    };

    // Initial check after 5 seconds
    const timeout = setTimeout(checkForUpdates, 5000);
    
    // Check every hour
    const interval = setInterval(checkForUpdates, 60 * 60 * 1000);

    return () => {
      clearTimeout(timeout);
      clearInterval(interval);
    };
  }, []);

  const handleUpdate = () => {
    // In production, trigger update process
    window.location.reload();
  };

  const handleDismiss = () => {
    setUpdateAvailable(false);
    localStorage.setItem('bul_update_dismissed', Date.now().toString());
  };

  if (!updateAvailable) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ y: 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 100, opacity: 0 }}
        className="fixed bottom-4 right-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-4 max-w-sm z-50"
      >
        <div className="flex items-start gap-3">
          <FiCheckCircle size={24} className="text-green-500 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-1">
              Actualización Disponible
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Hay una nueva versión disponible. Actualiza para obtener las últimas características.
            </p>
            <div className="flex gap-2">
              <button
                onClick={handleUpdate}
                className="btn btn-primary text-sm flex-1"
              >
                <FiDownload size={16} className="mr-1" />
                Actualizar
              </button>
              <button
                onClick={handleDismiss}
                className="btn btn-secondary text-sm"
              >
                <FiX size={16} />
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


