'use client';

import { useState } from 'react';
import { Activity, User, Robot, Settings, Clock } from 'lucide-react';

interface ActivityItem {
  id: string;
  type: 'user' | 'robot' | 'system';
  action: string;
  timestamp: Date;
  user?: string;
}

export default function ActivityFeed() {
  const [activities] = useState<ActivityItem[]>([
    {
      id: '1',
      type: 'user',
      action: 'Movió el robot a posición (0.5, 0.3, 0.2)',
      timestamp: new Date(),
      user: 'Admin',
    },
    {
      id: '2',
      type: 'robot',
      action: 'Completó movimiento exitosamente',
      timestamp: new Date(Date.now() - 60000),
    },
    {
      id: '3',
      type: 'system',
      action: 'Backup automático completado',
      timestamp: new Date(Date.now() - 120000),
    },
    {
      id: '4',
      type: 'user',
      action: 'Actualizó configuración de seguridad',
      timestamp: new Date(Date.now() - 180000),
      user: 'Operador',
    },
    {
      id: '5',
      type: 'robot',
      action: 'Detectado límite de velocidad',
      timestamp: new Date(Date.now() - 240000),
    },
  ]);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'user':
        return <User className="w-4 h-4 text-blue-400" />;
      case 'robot':
        return <Robot className="w-4 h-4 text-green-400" />;
      default:
        return <Settings className="w-4 h-4 text-gray-400" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'user':
        return 'bg-blue-500/10 border-blue-500/50';
      case 'robot':
        return 'bg-green-500/10 border-green-500/50';
      default:
        return 'bg-gray-700/50 border-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Activity className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Feed de Actividad</h3>
        </div>

        {/* Activities List */}
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {activities.map((activity) => (
            <div
              key={activity.id}
              className={`p-4 rounded-lg border ${getTypeColor(activity.type)}`}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5">{getTypeIcon(activity.type)}</div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="text-sm text-white">{activity.action}</p>
                    {activity.user && (
                      <span className="text-xs text-gray-400">por {activity.user}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-gray-400">
                    <Clock className="w-3 h-3" />
                    {activity.timestamp.toLocaleString('es-ES')}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {activities.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No hay actividad reciente</p>
          </div>
        )}
      </div>
    </div>
  );
}


