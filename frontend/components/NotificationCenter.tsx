'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiBell, FiX, FiCheckCircle, FiAlertCircle, FiInfo, FiAlertTriangle } from 'react-icons/fi';
import { format } from 'date-fns';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export default function NotificationCenter() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    // Load notifications from localStorage
    const loadNotifications = () => {
      const stored = localStorage.getItem('bul_notifications');
      if (stored) {
        try {
          setNotifications(JSON.parse(stored).map((n: any) => ({
            ...n,
            timestamp: new Date(n.timestamp),
          })));
        } catch (error) {
          console.error('Error loading notifications:', error);
        }
      }
    };

    loadNotifications();

    // Listen for new notifications (storage event for cross-tab sync)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'bul_notifications') {
        loadNotifications();
      }
    };

    // Listen for custom events (same tab)
    const handleCustomStorage = () => {
      loadNotifications();
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('bul_notification_update', handleCustomStorage);
    
    // Poll for updates
    const interval = setInterval(loadNotifications, 1000);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('bul_notification_update', handleCustomStorage);
      clearInterval(interval);
    };
  }, []);

  const unreadCount = notifications.filter((n) => !n.read).length;

  const markAsRead = (id: string) => {
    const updated = notifications.map((n) =>
      n.id === id ? { ...n, read: true } : n
    );
    setNotifications(updated);
    localStorage.setItem('bul_notifications', JSON.stringify(updated));
  };

  const markAllAsRead = () => {
    const updated = notifications.map((n) => ({ ...n, read: true }));
    setNotifications(updated);
    localStorage.setItem('bul_notifications', JSON.stringify(updated));
  };

  const removeNotification = (id: string) => {
    const updated = notifications.filter((n) => n.id !== id);
    setNotifications(updated);
    localStorage.setItem('bul_notifications', JSON.stringify(updated));
  };

  const clearAll = () => {
    setNotifications([]);
    localStorage.setItem('bul_notifications', JSON.stringify([]));
  };

  const getIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success':
        return <FiCheckCircle className="text-green-500" size={20} />;
      case 'error':
        return <FiAlertCircle className="text-red-500" size={20} />;
      case 'warning':
        return <FiAlertTriangle className="text-yellow-500" size={20} />;
      default:
        return <FiInfo className="text-blue-500" size={20} />;
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="btn-icon relative"
        title="Notificaciones"
      >
        <FiBell size={20} />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            <div
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              className="absolute right-0 top-full mt-2 w-96 bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 z-50 max-h-[500px] overflow-hidden flex flex-col"
            >
              <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Notificaciones {unreadCount > 0 && `(${unreadCount})`}
                </h3>
                <div className="flex items-center gap-2">
                  {unreadCount > 0 && (
                    <button
                      onClick={markAllAsRead}
                      className="text-sm text-primary-600 hover:text-primary-700"
                    >
                      Marcar todas
                    </button>
                  )}
                  <button onClick={() => setIsOpen(false)} className="btn-icon">
                    <FiX size={18} />
                  </button>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto">
                {notifications.length === 0 ? (
                  <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                    <FiBell size={48} className="mx-auto mb-2 opacity-50" />
                    <p>No hay notificaciones</p>
                  </div>
                ) : (
                  <div className="divide-y divide-gray-200 dark:divide-gray-700">
                    {notifications.map((notification) => (
                      <motion.div
                        key={notification.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className={`p-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${
                          !notification.read ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          {getIcon(notification.type)}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex-1">
                                <h4 className="font-medium text-gray-900 dark:text-white text-sm">
                                  {notification.title}
                                </h4>
                                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                                  {notification.message}
                                </p>
                                <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
                                  {format(notification.timestamp, 'PPp')}
                                </p>
                                {notification.action && (
                                  <button
                                    onClick={notification.action.onClick}
                                    className="mt-2 text-sm text-primary-600 hover:text-primary-700"
                                  >
                                    {notification.action.label}
                                  </button>
                                )}
                              </div>
                              <div className="flex items-center gap-1">
                                {!notification.read && (
                                  <div className="w-2 h-2 bg-primary-500 rounded-full" />
                                )}
                                <button
                                  onClick={() => removeNotification(notification.id)}
                                  className="btn-icon text-gray-400 hover:text-gray-600"
                                >
                                  <FiX size={16} />
                                </button>
                              </div>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </div>

              {notifications.length > 0 && (
                <div className="p-3 border-t border-gray-200 dark:border-gray-700">
                  <button
                    onClick={clearAll}
                    className="w-full text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
                  >
                    Limpiar todas
                  </button>
                </div>
              )}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

