'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiBell, FiX, FiCheckCircle, FiAlertCircle, FiInfo, FiAlertTriangle } from 'react-icons/fi';
import { toastManager, Toast as ToastType } from '@/lib/toast-manager';
import { cn } from '@/utils/classNames';
import { Badge } from './Badge';
import { Button } from './Button';

const icons = {
  success: FiCheckCircle,
  error: FiAlertCircle,
  warning: FiAlertTriangle,
  info: FiInfo,
};

const styles = {
  success: 'text-green-600 dark:text-green-400',
  error: 'text-red-600 dark:text-red-400',
  warning: 'text-yellow-600 dark:text-yellow-400',
  info: 'text-blue-600 dark:text-blue-400',
};

export function NotificationCenter() {
  const [isOpen, setIsOpen] = useState(false);
  const [notifications, setNotifications] = useState<ToastType[]>([]);

  useEffect(() => {
    const unsubscribe = toastManager.subscribe((toasts) => {
      setNotifications(toasts);
    });
    return unsubscribe;
  }, []);

  const unreadCount = notifications.length;

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsOpen(!isOpen)}
        className="relative"
        aria-label="Notificaciones"
      >
        <FiBell size={20} />
        {unreadCount > 0 && (
          <Badge
            variant="error"
            className="absolute -top-1 -right-1 min-w-[18px] h-[18px] flex items-center justify-center text-xs px-1"
          >
            {unreadCount > 99 ? '99+' : unreadCount}
          </Badge>
        )}
      </Button>

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
              className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl z-50 max-h-96 overflow-hidden flex flex-col"
            >
              <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Notificaciones
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => toastManager.clear()}
                  className="text-xs"
                >
                  Limpiar
                </Button>
              </div>

              <div className="flex-1 overflow-y-auto">
                {notifications.length === 0 ? (
                  <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                    No hay notificaciones
                  </div>
                ) : (
                  <div className="divide-y divide-gray-200 dark:divide-gray-700">
                    {notifications.map((notification) => {
                      const Icon = icons[notification.type];
                      return (
                        <div
                          key={notification.id}
                          className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                        >
                          <div className="flex items-start gap-3">
                            <Icon
                              size={20}
                              className={cn('flex-shrink-0 mt-0.5', styles[notification.type])}
                            />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-gray-900 dark:text-white">
                                {notification.message}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                {new Date(notification.timestamp).toLocaleTimeString()}
                              </p>
                            </div>
                            <button
                              onClick={() => toastManager.remove(notification.id)}
                              className="flex-shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                              aria-label="Cerrar"
                            >
                              <FiX size={16} />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

