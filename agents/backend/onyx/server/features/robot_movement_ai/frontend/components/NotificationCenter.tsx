'use client';

import { useState, useEffect } from 'react';
import { Bell, X, CheckCircle, AlertCircle, Info, XCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import { format } from 'date-fns';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
}

export default function NotificationCenter() {
  const [notifications, setNotifications] = useState<Notification[]>([
    {
      id: '1',
      type: 'success',
      title: 'Robot conectado',
      message: 'El robot se ha conectado exitosamente',
      timestamp: new Date(),
      read: false,
    },
    {
      id: '2',
      type: 'warning',
      title: 'Velocidad alta',
      message: 'La velocidad del robot está por encima del límite recomendado',
      timestamp: new Date(Date.now() - 60000),
      read: false,
    },
  ]);
  const [filter, setFilter] = useState<'all' | 'unread' | 'success' | 'error' | 'warning' | 'info'>('all');

  const getIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-600" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-600" />;
      case 'info':
        return <Info className="w-5 h-5 text-tesla-blue" />;
      default:
        return <Bell className="w-5 h-5 text-tesla-gray-dark" />;
    }
  };

  const getColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'border-green-200 bg-green-50';
      case 'error':
        return 'border-red-200 bg-red-50';
      case 'warning':
        return 'border-yellow-200 bg-yellow-50';
      case 'info':
        return 'border-blue-200 bg-blue-50';
      default:
        return 'border-gray-200 bg-gray-50';
    }
  };

  const filteredNotifications = notifications.filter((notif) => {
    if (filter === 'all') return true;
    if (filter === 'unread') return !notif.read;
    return notif.type === filter;
  });

  const unreadCount = notifications.filter((n) => !n.read).length;

  const markAsRead = (id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  };

  const markAllAsRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
    toast.success('Todas las notificaciones marcadas como leídas');
  };

  const deleteNotification = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const clearAll = () => {
    setNotifications([]);
    toast.success('Todas las notificaciones eliminadas');
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-tesla-blue" />
            <h3 className="text-lg font-semibold text-tesla-black">Centro de Notificaciones</h3>
            {unreadCount > 0 && (
              <span className="px-2.5 py-1 bg-red-600 text-white text-xs font-semibold rounded-full">
                {unreadCount}
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={markAllAsRead}
              disabled={unreadCount === 0}
              className="px-4 py-2 bg-tesla-blue hover:bg-opacity-90 text-white rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium min-h-[44px]"
              aria-label="Marcar todas las notificaciones como leídas"
            >
              Marcar todas como leídas
            </button>
            <button
              onClick={clearAll}
              className="px-4 py-2 bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all font-medium min-h-[44px]"
              aria-label="Limpiar todas las notificaciones"
            >
              Limpiar todo
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {(['all', 'unread', 'success', 'error', 'warning', 'info'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all min-h-[44px] ${
                filter === f
                  ? 'bg-tesla-blue text-white shadow-sm'
                  : 'bg-white border-2 border-gray-300 text-tesla-black hover:border-gray-400'
              }`}
            >
              {f === 'all' ? 'Todas' : f === 'unread' ? 'No leídas' : f}
            </button>
          ))}
        </div>

        {/* Notifications List */}
        <div className="space-y-3 max-h-[600px] overflow-y-auto scrollbar-hide">
          {filteredNotifications.length === 0 ? (
            <div className="text-center py-12 text-tesla-gray-dark">
              <Bell className="w-12 h-12 mx-auto mb-4 text-tesla-gray-light opacity-50" />
              <p className="font-medium">No hay notificaciones</p>
            </div>
          ) : (
            filteredNotifications.map((notif) => (
              <div
                key={notif.id}
                className={`p-4 rounded-md border shadow-sm transition-all hover:shadow-md ${getColor(notif.type)} ${
                  !notif.read ? 'ring-2 ring-tesla-blue/30' : ''
                }`}
              >
                <div className="flex items-start gap-3">
                  {getIcon(notif.type)}
                  <div className="flex-1">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <h4 className={`font-semibold mb-1 ${
                          notif.type === 'success' ? 'text-green-700' :
                          notif.type === 'error' ? 'text-red-700' :
                          notif.type === 'warning' ? 'text-yellow-700' :
                          'text-tesla-blue'
                        }`}>{notif.title}</h4>
                        <p className={`text-sm ${
                          notif.type === 'success' ? 'text-green-600' :
                          notif.type === 'error' ? 'text-red-600' :
                          notif.type === 'warning' ? 'text-yellow-600' :
                          'text-tesla-gray-dark'
                        }`}>{notif.message}</p>
                        <p className="text-xs text-tesla-gray-dark mt-2">
                          {format(notif.timestamp, 'dd/MM/yyyy HH:mm:ss')}
                        </p>
                      </div>
                      <div className="flex gap-1">
                        {!notif.read && (
                          <button
                            onClick={() => markAsRead(notif.id)}
                            className="p-2 text-tesla-gray-dark hover:text-tesla-blue transition-colors rounded hover:bg-gray-100 min-h-[44px] min-w-[44px] flex items-center justify-center"
                            title="Marcar como leída"
                            aria-label="Marcar notificación como leída"
                          >
                            <CheckCircle className="w-4 h-4" />
                          </button>
                        )}
                        <button
                          onClick={() => deleteNotification(notif.id)}
                          className="p-2 text-tesla-gray-dark hover:text-red-600 transition-colors rounded hover:bg-gray-100 min-h-[44px] min-w-[44px] flex items-center justify-center"
                          title="Eliminar"
                          aria-label="Eliminar notificación"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}


