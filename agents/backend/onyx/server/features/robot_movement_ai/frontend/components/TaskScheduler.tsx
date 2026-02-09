'use client';

import { useState } from 'react';
import { Calendar, Plus, Play, Pause, Trash2, Clock } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Task {
  id: string;
  name: string;
  description: string;
  schedule: string;
  enabled: boolean;
  lastRun?: Date;
  nextRun?: Date;
}

export default function TaskScheduler() {
  const [tasks, setTasks] = useState<Task[]>([
    {
      id: '1',
      name: 'Limpieza de Logs',
      description: 'Eliminar logs antiguos',
      schedule: '0 2 * * *',
      enabled: true,
      lastRun: new Date(Date.now() - 86400000),
      nextRun: new Date(Date.now() + 3600000),
    },
    {
      id: '2',
      name: 'Backup Automático',
      description: 'Crear backup del sistema',
      schedule: '0 3 * * *',
      enabled: true,
      lastRun: new Date(Date.now() - 172800000),
      nextRun: new Date(Date.now() + 7200000),
    },
    {
      id: '3',
      name: 'Actualización de Métricas',
      description: 'Actualizar métricas del sistema',
      schedule: '*/5 * * * *',
      enabled: false,
    },
  ]);
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDescription, setNewDescription] = useState('');
  const [newSchedule, setNewSchedule] = useState('0 0 * * *');

  const handleAdd = () => {
    if (!newName.trim()) {
      toast.error('El nombre de la tarea es requerido');
      return;
    }

    const newTask: Task = {
      id: Date.now().toString(),
      name: newName,
      description: newDescription,
      schedule: newSchedule,
      enabled: true,
    };

    setTasks([...tasks, newTask]);
    setNewName('');
    setNewDescription('');
    setNewSchedule('0 0 * * *');
    setShowAdd(false);
    toast.success('Tarea creada');
  };

  const handleToggle = (id: string) => {
    setTasks((prev) =>
      prev.map((t) =>
        t.id === id ? { ...t, enabled: !t.enabled } : t
      )
    );
    toast.success('Tarea actualizada');
  };

  const handleDelete = (id: string) => {
    setTasks(tasks.filter((t) => t.id !== id));
    toast.success('Tarea eliminada');
  };

  const handleRunNow = (id: string) => {
    setTasks((prev) =>
      prev.map((t) =>
        t.id === id ? { ...t, lastRun: new Date() } : t
      )
    );
    toast.info('Ejecutando tarea...');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Programador de Tareas</h3>
          </div>
          <button
            onClick={() => setShowAdd(!showAdd)}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Nueva Tarea
          </button>
        </div>

        {/* Add Form */}
        {showAdd && (
          <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <h4 className="text-sm font-medium text-white mb-3">Crear Nueva Tarea</h4>
            <div className="space-y-3">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Nombre de la tarea"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <textarea
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                placeholder="Descripción"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                rows={2}
              />
              <input
                type="text"
                value={newSchedule}
                onChange={(e) => setNewSchedule(e.target.value)}
                placeholder="Cron schedule (ej: 0 2 * * *)"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500 font-mono text-sm"
              />
              <div className="flex gap-2">
                <button
                  onClick={handleAdd}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                >
                  Crear
                </button>
                <button
                  onClick={() => {
                    setShowAdd(false);
                    setNewName('');
                    setNewDescription('');
                    setNewSchedule('0 0 * * *');
                  }}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Cancelar
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Tasks List */}
        <div className="space-y-3">
          {tasks.map((task) => (
            <div
              key={task.id}
              className={`p-4 rounded-lg border ${
                task.enabled
                  ? 'bg-green-500/10 border-green-500/50'
                  : 'bg-gray-700/50 border-gray-600'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-white">{task.name}</h4>
                    <span className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded font-mono">
                      {task.schedule}
                    </span>
                  </div>
                  <p className="text-sm text-gray-300 mb-2">{task.description}</p>
                  {task.lastRun && (
                    <p className="text-xs text-gray-400 mb-1">
                      Última ejecución: {task.lastRun.toLocaleString('es-ES')}
                    </p>
                  )}
                  {task.nextRun && task.enabled && (
                    <p className="text-xs text-green-400">
                      Próxima ejecución: {task.nextRun.toLocaleString('es-ES')}
                    </p>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleRunNow(task.id)}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                  >
                    <Play className="w-3 h-3" />
                    Ejecutar
                  </button>
                  <button
                    onClick={() => handleToggle(task.id)}
                    className={`px-3 py-1 text-sm rounded-lg transition-colors flex items-center gap-2 ${
                      task.enabled
                        ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    {task.enabled ? (
                      <>
                        <Pause className="w-3 h-3" />
                        Pausar
                      </>
                    ) : (
                      <>
                        <Play className="w-3 h-3" />
                        Activar
                      </>
                    )}
                  </button>
                  <button
                    onClick={() => handleDelete(task.id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


