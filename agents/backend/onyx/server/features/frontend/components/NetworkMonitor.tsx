'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiWifi, FiWifiOff, FiActivity } from 'react-icons/fi';

interface NetworkRequest {
  id: string;
  url: string;
  method: string;
  status: number | null;
  duration: number;
  timestamp: Date;
}

export default function NetworkMonitor() {
  const [requests, setRequests] = useState<NetworkRequest[]>([]);
  const [isOnline, setIsOnline] = useState(true);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsOnline(navigator.onLine);

    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Intercept fetch requests
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const startTime = Date.now();
      const url = typeof args[0] === 'string' ? args[0] : args[0].url;
      const method = args[1]?.method || 'GET';

      try {
        const response = await originalFetch(...args);
        const duration = Date.now() - startTime;

        setRequests((prev) => [
          {
            id: Date.now().toString() + Math.random(),
            url,
            method,
            status: response.status,
            duration,
            timestamp: new Date(),
          },
          ...prev,
        ].slice(0, 20)); // Keep last 20

        return response;
      } catch (error) {
        const duration = Date.now() - startTime;
        setRequests((prev) => [
          {
            id: Date.now().toString() + Math.random(),
            url,
            method,
            status: null,
            duration,
            timestamp: new Date(),
          },
          ...prev,
        ].slice(0, 20));

        throw error;
      }
    };

    // Toggle with Ctrl+Shift+N
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'N') {
        setIsVisible((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.fetch = originalFetch;
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  if (!isVisible) return null;

  const recentRequests = requests.slice(0, 10);
  const avgDuration = requests.length > 0
    ? requests.reduce((sum, r) => sum + r.duration, 0) / requests.length
    : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="fixed top-20 right-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-4 z-50 min-w-[300px] max-w-md"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {isOnline ? (
            <FiWifi size={18} className="text-green-500" />
          ) : (
            <FiWifiOff size={18} className="text-red-500" />
          )}
          <h4 className="font-semibold text-gray-900 dark:text-white text-sm">Red</h4>
        </div>
        <button
          onClick={() => setIsVisible(false)}
          className="btn-icon text-gray-400"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6L6 18" />
            <path d="M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="space-y-2 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-gray-600 dark:text-gray-400">Estado</span>
          <span className={`font-medium ${isOnline ? 'text-green-600' : 'text-red-600'}`}>
            {isOnline ? 'En línea' : 'Sin conexión'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-600 dark:text-gray-400">Solicitudes</span>
          <span className="font-mono text-gray-900 dark:text-white">{requests.length}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-600 dark:text-gray-400">Tiempo Promedio</span>
          <span className="font-mono text-gray-900 dark:text-white">{Math.round(avgDuration)}ms</span>
        </div>
      </div>

      {recentRequests.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <FiActivity size={14} className="text-gray-400" />
            <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
              Solicitudes Recientes
            </span>
          </div>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {recentRequests.map((req) => (
              <div
                key={req.id}
                className="text-xs p-2 bg-gray-50 dark:bg-gray-700 rounded"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-gray-900 dark:text-white">
                    {req.method}
                  </span>
                  <div className="flex items-center gap-2">
                    {req.status && (
                      <span
                        className={`px-1.5 py-0.5 rounded ${
                          req.status >= 200 && req.status < 300
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            : req.status >= 400
                            ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                            : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                        }`}
                      >
                        {req.status}
                      </span>
                    )}
                    <span className="text-gray-500 dark:text-gray-400">
                      {req.duration}ms
                    </span>
                  </div>
                </div>
                <div className="text-gray-600 dark:text-gray-400 truncate" title={req.url}>
                  {req.url}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}


