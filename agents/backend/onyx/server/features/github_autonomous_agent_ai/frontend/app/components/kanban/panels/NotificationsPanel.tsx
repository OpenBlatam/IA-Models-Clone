'use client';

import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { cn } from '../../../utils/cn';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  timestamp: Date;
  read: boolean;
}

interface NotificationsPanelProps {
  notifications: Notification[];
  unreadCount: number;
  darkMode: boolean;
  onMarkAllAsRead: () => void;
  onClearAll: () => void;
  onClose: () => void;
}

export function NotificationsPanel({
  notifications,
  unreadCount,
  darkMode,
  onMarkAllAsRead,
  onClearAll,
  onClose,
}: NotificationsPanelProps) {
  return (
    <div className="notifications-panel absolute right-0 top-full mt-2 w-96 bg-white border border-gray-200 rounded-lg shadow-xl z-50 max-h-96 overflow-hidden flex flex-col">
      <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
        <h3 className="font-semibold text-sm">Notificaciones</h3>
        <div className="flex items-center gap-2">
          {unreadCount > 0 && (
            <button
              onClick={onMarkAllAsRead}
              className="text-xs text-blue-600 hover:text-blue-800"
            >
              Marcar todas como leídas
            </button>
          )}
          <button
            onClick={() => {
              onClearAll();
              onClose();
            }}
            className="text-xs text-red-600 hover:text-red-800"
          >
            Limpiar
          </button>
        </div>
      </div>
      <div className="overflow-y-auto flex-1">
        {notifications.length === 0 ? (
          <div className="px-4 py-8 text-center text-gray-500 text-sm">
            No hay notificaciones
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {notifications.map((notif) => (
              <div
                key={notif.id}
                className={cn(
                  "px-4 py-3 hover:bg-gray-50 transition-colors cursor-pointer",
                  !notif.read && "bg-blue-50"
                )}
              >
                <div className="flex items-start gap-3">
                  <div className={cn(
                    "w-2 h-2 rounded-full mt-2 flex-shrink-0",
                    notif.type === 'success' && "bg-green-500",
                    notif.type === 'error' && "bg-red-500",
                    notif.type === 'warning' && "bg-yellow-500",
                    notif.type === 'info' && "bg-blue-500"
                  )} />
                  <div className="flex-1 min-w-0">
                    <p className={cn(
                      "text-sm font-medium",
                      !notif.read && "font-semibold"
                    )}>
                      {notif.title}
                    </p>
                    {notif.message && (
                      <p className="text-xs text-gray-600 mt-1 truncate">
                        {notif.message}
                      </p>
                    )}
                    <p className="text-xs text-gray-400 mt-1">
                      {format(notif.timestamp, 'HH:mm', { locale: es })}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

