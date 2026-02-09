'use client';

import { useState, useEffect } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { Clock, Activity, Play, Square, Home, Zap } from 'lucide-react';
import { format } from 'date-fns';

interface ActivityEvent {
  id: string;
  type: 'move' | 'stop' | 'home' | 'command' | 'error' | 'connect' | 'disconnect';
  description: string;
  timestamp: Date;
  details?: any;
}

export default function ActivityTimeline() {
  const { status, chatMessages } = useRobotStore();
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [filter, setFilter] = useState<'all' | 'move' | 'stop' | 'command' | 'error'>('all');

  useEffect(() => {
    // Simulate activity events from robot status and chat
    const newEvents: ActivityEvent[] = [];

    if (status?.robot_status.connected) {
      newEvents.push({
        id: `connect-${Date.now()}`,
        type: 'connect',
        description: 'Robot conectado',
        timestamp: new Date(),
      });
    }

    if (status?.robot_status.moving) {
      newEvents.push({
        id: `move-${Date.now()}`,
        type: 'move',
        description: `Movimiento a posición (${status.robot_status.position?.x.toFixed(2)}, ${status.robot_status.position?.y.toFixed(2)}, ${status.robot_status.position?.z.toFixed(2)})`,
        timestamp: new Date(),
        details: status.robot_status.position,
      });
    }

    if (chatMessages.length > 0) {
      const lastMessage = chatMessages[chatMessages.length - 1];
      if (lastMessage.role === 'user') {
        newEvents.push({
          id: `cmd-${Date.now()}`,
          type: 'command',
          description: `Comando: ${lastMessage.content.substring(0, 50)}...`,
          timestamp: new Date(),
        });
      }
    }

    // Keep last 50 events
    setEvents((prev) => {
      const combined = [...newEvents, ...prev];
      return combined.slice(0, 50);
    });
  }, [status, chatMessages]);

  const getIcon = (type: string) => {
    switch (type) {
      case 'move':
        return <Play className="w-4 h-4 text-blue-400" />;
      case 'stop':
        return <Square className="w-4 h-4 text-red-400" />;
      case 'home':
        return <Home className="w-4 h-4 text-green-400" />;
      case 'command':
        return <Zap className="w-4 h-4 text-yellow-400" />;
      case 'error':
        return <Activity className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getColor = (type: string) => {
    switch (type) {
      case 'move':
        return 'border-blue-500/50 bg-blue-500/10';
      case 'stop':
        return 'border-red-500/50 bg-red-500/10';
      case 'home':
        return 'border-green-500/50 bg-green-500/10';
      case 'command':
        return 'border-yellow-500/50 bg-yellow-500/10';
      case 'error':
        return 'border-red-500/50 bg-red-500/10';
      default:
        return 'border-gray-500/50 bg-gray-500/10';
    }
  };

  const filteredEvents = events.filter((event) => {
    if (filter === 'all') return true;
    return event.type === filter;
  });

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Línea de Tiempo de Actividad</h3>
          </div>
          <div className="flex gap-2">
            {(['all', 'move', 'stop', 'command', 'error'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                  filter === f
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {f === 'all' ? 'Todas' : f}
              </button>
            ))}
          </div>
        </div>

        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-700"></div>

          {/* Events */}
          <div className="space-y-4">
            {filteredEvents.length === 0 ? (
              <div className="text-center py-12 text-gray-400">
                <Clock className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No hay actividad reciente</p>
              </div>
            ) : (
              filteredEvents.map((event, index) => (
                <div key={event.id} className="relative flex items-start gap-4">
                  {/* Icon */}
                  <div
                    className={`relative z-10 p-2 rounded-full border-2 ${getColor(event.type)}`}
                  >
                    {getIcon(event.type)}
                  </div>

                  {/* Content */}
                  <div className="flex-1 pb-4">
                    <div className={`p-4 rounded-lg border ${getColor(event.type)}`}>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <p className="text-white font-medium mb-1">{event.description}</p>
                          <p className="text-xs text-gray-400">
                            {format(event.timestamp, 'dd/MM/yyyy HH:mm:ss')}
                          </p>
                          {event.details && (
                            <details className="mt-2">
                              <summary className="text-xs text-gray-400 cursor-pointer">
                                Ver detalles
                              </summary>
                              <pre className="mt-2 text-xs text-gray-400 bg-gray-900/50 p-2 rounded overflow-x-auto">
                                {JSON.stringify(event.details, null, 2)}
                              </pre>
                            </details>
                          )}
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
    </div>
  );
}


