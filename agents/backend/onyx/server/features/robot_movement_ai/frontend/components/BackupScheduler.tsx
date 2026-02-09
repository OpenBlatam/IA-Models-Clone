'use client';

import { useState } from 'react';
import { Calendar, Clock, Play, Pause, Trash2 } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface BackupSchedule {
  id: string;
  name: string;
  frequency: 'daily' | 'weekly' | 'monthly';
  time: string;
  enabled: boolean;
  lastRun?: Date;
  nextRun?: Date;
}

export default function BackupScheduler() {
  const [schedules, setSchedules] = useState<BackupSchedule[]>([
    {
      id: '1',
      name: 'Backup Diario',
      frequency: 'daily',
      time: '02:00',
      enabled: true,
      lastRun: new Date(Date.now() - 86400000),
      nextRun: new Date(Date.now() + 3600000),
    },
    {
      id: '2',
      name: 'Backup Semanal',
      frequency: 'weekly',
      time: '03:00',
      enabled: false,
    },
  ]);

  const handleToggle = (id: string) => {
    setSchedules((prev) =>
      prev.map((s) =>
        s.id === id ? { ...s, enabled: !s.enabled } : s
      )
    );
    toast.success('Programación actualizada');
  };

  const handleDelete = (id: string) => {
    setSchedules((prev) => prev.filter((s) => s.id !== id));
    toast.success('Programación eliminada');
  };

  const handleRunNow = (id: string) => {
    toast.info('Ejecutando backup...');
    setTimeout(() => {
      setSchedules((prev) =>
        prev.map((s) =>
          s.id === id ? { ...s, lastRun: new Date() } : s
        )
      );
      toast.success('Backup completado');
    }, 2000);
  };

  const getFrequencyLabel = (freq: string) => {
    switch (freq) {
      case 'daily':
        return 'Diario';
      case 'weekly':
        return 'Semanal';
      case 'monthly':
        return 'Mensual';
      default:
        return freq;
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Calendar className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Programador de Backups</h3>
        </div>

        {/* Schedules List */}
        <div className="space-y-4">
          {schedules.map((schedule) => (
            <div
              key={schedule.id}
              className={`p-4 rounded-lg border ${
                schedule.enabled
                  ? 'bg-green-500/10 border-green-500/50'
                  : 'bg-gray-700/50 border-gray-600'
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Clock className="w-4 h-4 text-primary-400" />
                    <h4 className="font-semibold text-white">{schedule.name}</h4>
                  </div>
                  <div className="text-sm text-gray-300 space-y-1">
                    <p>Frecuencia: {getFrequencyLabel(schedule.frequency)}</p>
                    <p>Hora: {schedule.time}</p>
                    {schedule.lastRun && (
                      <p className="text-xs text-gray-400">
                        Última ejecución: {schedule.lastRun.toLocaleString('es-ES')}
                      </p>
                    )}
                    {schedule.nextRun && schedule.enabled && (
                      <p className="text-xs text-green-400">
                        Próxima ejecución: {schedule.nextRun.toLocaleString('es-ES')}
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleRunNow(schedule.id)}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                  >
                    <Play className="w-3 h-3" />
                    Ejecutar Ahora
                  </button>
                  <button
                    onClick={() => handleToggle(schedule.id)}
                    className={`px-3 py-1 text-sm rounded-lg transition-colors flex items-center gap-2 ${
                      schedule.enabled
                        ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    {schedule.enabled ? (
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
                    onClick={() => handleDelete(schedule.id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {schedules.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <Calendar className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No hay programaciones configuradas</p>
          </div>
        )}
      </div>
    </div>
  );
}


