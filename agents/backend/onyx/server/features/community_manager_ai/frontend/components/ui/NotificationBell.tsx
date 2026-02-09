'use client';

import { Bell } from 'lucide-react';
import { useState } from 'react';
import { Badge } from './Badge';
import * as Popover from '@radix-ui/react-popover';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface Notification {
  id: string;
  title: string;
  message: string;
  time: string;
  read?: boolean;
  type?: 'info' | 'success' | 'warning' | 'error';
}

interface NotificationBellProps {
  notifications?: Notification[];
  onMarkAsRead?: (id: string) => void;
  onMarkAllAsRead?: () => void;
}

export const NotificationBell = ({
  notifications = [],
  onMarkAsRead,
  onMarkAllAsRead,
}: NotificationBellProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const unreadCount = notifications.filter((n) => !n.read).length;

  const handleMarkAsRead = (id: string) => {
    onMarkAsRead?.(id);
  };

  const handleMarkAllAsRead = () => {
    onMarkAllAsRead?.();
  };

  return (
    <Popover.Root open={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          className="relative rounded-lg p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          aria-label="Notificaciones"
          tabIndex={0}
        >
          <Bell className="h-5 w-5" />
          {unreadCount > 0 && (
            <span className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-xs font-bold text-white">
              {unreadCount > 9 ? '9+' : unreadCount}
            </span>
          )}
        </button>
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content
          className={cn(
            'z-50 w-80 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg',
            'animate-in fade-in-0 zoom-in-95'
          )}
          sideOffset={5}
          align="end"
        >
          <div className="p-4">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Notificaciones
              </h3>
              {unreadCount > 0 && (
                <button
                  type="button"
                  onClick={handleMarkAllAsRead}
                  className="text-sm text-primary-600 dark:text-primary-400 hover:underline"
                >
                  Marcar todas como leídas
                </button>
              )}
            </div>

            <div className="max-h-96 space-y-2 overflow-y-auto">
              <AnimatePresence>
                {notifications.length === 0 ? (
                  <div className="py-8 text-center text-sm text-gray-500 dark:text-gray-400">
                    No hay notificaciones
                  </div>
                ) : (
                  notifications.map((notification) => (
                    <motion.div
                      key={notification.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 10 }}
                      className={cn(
                        'cursor-pointer rounded-lg border p-3 transition-colors',
                        notification.read
                          ? 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50'
                          : 'border-primary-200 dark:border-primary-800 bg-primary-50 dark:bg-primary-900/20',
                        'hover:bg-gray-100 dark:hover:bg-gray-800'
                      )}
                      onClick={() => handleMarkAsRead(notification.id)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <p
                            className={cn(
                              'text-sm font-medium',
                              notification.read
                                ? 'text-gray-600 dark:text-gray-400'
                                : 'text-gray-900 dark:text-gray-100'
                            )}
                          >
                            {notification.title}
                          </p>
                          <p className="mt-1 text-xs text-gray-500 dark:text-gray-500">
                            {notification.message}
                          </p>
                          <p className="mt-1 text-xs text-gray-400 dark:text-gray-600">
                            {notification.time}
                          </p>
                        </div>
                        {!notification.read && (
                          <div className="ml-2 h-2 w-2 rounded-full bg-primary-600 dark:bg-primary-400" />
                        )}
                      </div>
                    </motion.div>
                  ))
                )}
              </AnimatePresence>
            </div>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
};



