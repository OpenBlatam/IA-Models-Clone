'use client';

import { useState, useEffect } from 'react';
import { Bell, X, CheckCircle, AlertCircle, Info } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';

export function NotificationCenter() {
  const [isOpen, setIsOpen] = useState(false);
  const [userId] = useState('user123');

  const { data: notifications } = useQuery({
    queryKey: ['notifications', userId],
    queryFn: () => musicApiService.getNotifications?.(userId) || Promise.resolve({ notifications: [] }),
    refetchInterval: 30000,
  });

  const unreadCount = notifications?.notifications?.filter((n: any) => !n.read).length || 0;

  const getIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Info className="w-5 h-5 text-blue-400" />;
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors text-white"
      >
        <Bell className="w-6 h-6" />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-xs text-white">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="absolute right-0 top-12 w-80 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 shadow-xl z-50 max-h-96 overflow-y-auto">
          <div className="p-4 border-b border-white/20 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">Notificaciones</h3>
            <button
              onClick={() => setIsOpen(false)}
              className="text-gray-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="p-2">
            {notifications?.notifications?.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <Bell className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No hay notificaciones</p>
              </div>
            ) : (
              <div className="space-y-2">
                {notifications?.notifications?.map((notif: any, idx: number) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg ${
                      !notif.read ? 'bg-purple-500/20 border border-purple-400/30' : 'bg-white/5'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {getIcon(notif.type || 'info')}
                      <div className="flex-1">
                        <p className="text-sm font-medium text-white">{notif.title}</p>
                        <p className="text-xs text-gray-300 mt-1">{notif.message}</p>
                        {notif.timestamp && (
                          <p className="text-xs text-gray-400 mt-1">
                            {new Date(notif.timestamp).toLocaleString()}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

