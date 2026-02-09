'use client';

import { useState } from 'react';
import { Wrench, CheckCircle, AlertTriangle, Clock, RefreshCw } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface MaintenanceTask {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'in-progress' | 'completed' | 'failed';
  lastRun?: Date;
  nextRun?: Date;
  interval: number; // in hours
}

export default function MaintenancePanel() {
  const [tasks, setTasks] = useState<MaintenanceTask[]>([
    {
      id: '1',
      name: 'Limpieza de logs',
      description: 'Eliminar logs antiguos (>30 días)',
      status: 'pending',
      interval: 24,
    },
    {
      id: '2',
      name: 'Verificación de sistema',
      description: 'Verificar integridad del sistema',
      status: 'pending',
      interval: 12,
    },
    {
      id: '3',
      name: 'Optimización de base de datos',
      description: 'Optimizar índices y limpiar datos',
      status: 'pending',
      interval: 168, // weekly
    },
    {
      id: '4',
      name: 'Backup automático',
      description: 'Crear backup del sistema',
      status: 'completed',
      lastRun: new Date(Date.now() - 3600000),
      nextRun: new Date(Date.now() + 23 * 3600000),
      interval: 24,
    },
  ]);

  const handleRunTask = async (taskId: string) => {
    setTasks((prev) =>
      prev.map((t) => (t.id === taskId ? { ...t, status: 'in-progress' as const } : t))
    );

    try {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      setTasks((prev) =>
        prev.map((t) =>
          t.id === taskId
            ? {
                ...t,
                status: 'completed' as const,
                lastRun: new Date(),
                nextRun: new Date(Date.now() + t.interval * 3600000),
              }
            : t
        )
      );
      toast.success('Tarea completada');
    } catch (error) {
      setTasks((prev) =>
        prev.map((t) => (t.id === taskId ? { ...t, status: 'failed' as const } : t))
      );
      toast.error('Error al ejecutar tarea');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'in-progress':
        return <RefreshCw className="w-5 h-5 text-blue-400 animate-spin" />;
      case 'failed':
        return <AlertTriangle className="w-5 h-5 text-red-400" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'border-green-500/50 bg-green-500/10';
      case 'in-progress':
        return 'border-blue-500/50 bg-blue-500/10';
      case 'failed':
        return 'border-red-500/50 bg-red-500/10';
      default:
        return 'border-gray-500/50 bg-gray-500/10';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Wrench className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Panel de Mantenimiento</h3>
        </div>

        <div className="space-y-3">
          {tasks.map((task) => (
            <div
              key={task.id}
              className={`p-4 rounded-lg border ${getStatusColor(task.status)}`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3 flex-1">
                  {getStatusIcon(task.status)}
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{task.name}</h4>
                    <p className="text-sm text-gray-300 mb-2">{task.description}</p>
                    <div className="flex gap-4 text-xs text-gray-400">
                      <span>Intervalo: {task.interval}h</span>
                      {task.lastRun && (
                        <span>
                          Última ejecución:{' '}
                          {task.lastRun.toLocaleString('es-ES', {
                            day: '2-digit',
                            month: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </span>
                      )}
                      {task.nextRun && (
                        <span>
                          Próxima:{' '}
                          {task.nextRun.toLocaleString('es-ES', {
                            day: '2-digit',
                            month: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => handleRunTask(task.id)}
                  disabled={task.status === 'in-progress'}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {task.status === 'in-progress' ? 'Ejecutando...' : 'Ejecutar'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


