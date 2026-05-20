'use client';

import React, { useState } from 'react';
import { Bell } from 'lucide-react';
import { clsx } from 'clsx';
import { Badge } from './Badge';
import { Dropdown } from './Dropdown';

interface Notification {
  id: string;
  title: string;
  message: string;
  time: string;
  read: boolean;
  type?: 'info' | 'success' | 'warning' | 'error';
}

interface NotificationBellProps {
  notifications: Notification[];
  unreadCount?: number;
  onNotificationClick?: (notification: Notification) => void;
  onMarkAllRead?: () => void;
  className?: string;
}

export const NotificationBell: React.FC<NotificationBellProps> = ({
  notifications,
  unreadCount = 0,
  onNotificationClick,
  onMarkAllRead,
  className,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const unreadNotifications = notifications.filter((n) => !n.read);

  const dropdownItems = [
    ...notifications.map((notification) => ({
      label: (
        <div
          className={clsx(
            'p-3 rounded-lg',
            !notification.read && 'bg-primary-50 dark:bg-primary-900/20'
          )}
        >
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="font-medium text-sm text-gray-900 dark:text-white">
                {notification.title}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                {notification.message}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                {notification.time}
              </p>
            </div>
            {!notification.read && (
              <div className="ml-2 w-2 h-2 bg-primary-600 rounded-full" />
            )}
          </div>
        </div>
      ),
      onClick: () => {
        onNotificationClick?.(notification);
        setIsOpen(false);
      },
    })),
    notifications.length > 0 && unreadCount > 0
      ? {
          separator: true,
          label: (
            <button
              onClick={() => {
                onMarkAllRead?.();
                setIsOpen(false);
              }}
              className="w-full text-left px-4 py-2 text-sm text-primary-600 dark:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              Marcar todas como leídas
            </button>
          ),
        }
      : null,
  ].filter(Boolean);

  return (
    <div className={clsx('relative', className)}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
        aria-label="Notificaciones"
      >
        <Bell className="h-6 w-6" />
        {unreadCount > 0 && (
          <Badge
            variant="danger"
            className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center p-0 text-xs"
          >
            {unreadCount > 9 ? '9+' : unreadCount}
          </Badge>
        )}
      </button>
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 z-50 max-h-96 overflow-y-auto">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white">
              Notificaciones
            </h3>
          </div>
          {notifications.length === 0 ? (
            <div className="p-8 text-center text-gray-500 dark:text-gray-400">
              No hay notificaciones
            </div>
          ) : (
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {notifications.map((notification) => (
                <div
                  key={notification.id}
                  onClick={() => {
                    onNotificationClick?.(notification);
                    setIsOpen(false);
                  }}
                  className={clsx(
                    'p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors',
                    !notification.read && 'bg-primary-50 dark:bg-primary-900/20'
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="font-medium text-sm text-gray-900 dark:text-white">
                        {notification.title}
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        {notification.message}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                        {notification.time}
                      </p>
                    </div>
                    {!notification.read && (
                      <div className="ml-2 w-2 h-2 bg-primary-600 rounded-full" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
          {notifications.length > 0 && unreadCount > 0 && (
            <div className="p-2 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => {
                  onMarkAllRead?.();
                  setIsOpen(false);
                }}
                className="w-full text-center px-4 py-2 text-sm text-primary-600 dark:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                Marcar todas como leídas
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};


