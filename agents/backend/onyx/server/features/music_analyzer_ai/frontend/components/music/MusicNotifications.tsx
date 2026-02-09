'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Bell, CheckCircle, XCircle, AlertCircle, Info } from 'lucide-react';
import toast from 'react-hot-toast';

export function MusicNotifications() {
  const [userId] = useState('user123');

  const { data: notifications } = useQuery({
    queryKey: ['notifications', userId],
    queryFn: () => musicApiService.getNotifications?.(userId, false, 50) || Promise.resolve({ notifications: [] }),
    refetchInterval: 30000,
  });

  const notificationList = notifications?.notifications || [];

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-400" />;
      default:
        return <Info className="w-5 h-5 text-blue-400" />;
    }
  };

  const getNotificationColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'bg-green-500/20 border-green-500/30';
      case 'error':
        return 'bg-red-500/20 border-red-500/30';
      case 'warning':
        return 'bg-yellow-500/20 border-yellow-500/30';
      default:
        return 'bg-blue-500/20 border-blue-500/30';
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Bell className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Notificaciones</h3>
        {notificationList.length > 0 && (
          <span className="px-2 py-1 bg-purple-600 text-white text-xs rounded-full">
            {notificationList.filter((n: any) => !n.read).length}
          </span>
        )}
      </div>

      {notificationList.length === 0 ? (
        <div className="text-center py-12">
          <Bell className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">No hay notificaciones</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {notificationList.map((notification: any, idx: number) => (
            <div
              key={notification.id || idx}
              className={`p-4 rounded-lg border ${
                notification.read ? 'bg-white/5' : getNotificationColor(notification.type || 'info')
              }`}
            >
              <div className="flex items-start gap-3">
                {getNotificationIcon(notification.type || 'info')}
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium">{notification.title || 'Notificación'}</p>
                  <p className="text-sm text-gray-300 mt-1">{notification.message || ''}</p>
                  <p className="text-xs text-gray-400 mt-2">
                    {notification.created_at
                      ? new Date(notification.created_at).toLocaleString()
                      : 'Fecha desconocida'}
                  </p>
                </div>
                {!notification.read && (
                  <div className="w-2 h-2 bg-purple-400 rounded-full flex-shrink-0 mt-2" />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

