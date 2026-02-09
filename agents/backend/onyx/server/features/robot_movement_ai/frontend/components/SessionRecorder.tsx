'use client';

import { useState, useEffect } from 'react';
import { useLocalStorage } from '@/lib/hooks/useLocalStorage';
import { Video, Play, Square, Download, Trash2 } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import { format } from 'date-fns';

interface Session {
  id: string;
  name: string;
  startTime: Date;
  endTime?: Date;
  duration: number;
  actions: any[];
}

export default function SessionRecorder() {
  const [sessions, setSessions] = useLocalStorage<Session[]>('sessions', []);
  const [isRecording, setIsRecording] = useState(false);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [actions, setActions] = useState<any[]>([]);

  const handleStartRecording = () => {
    const session: Session = {
      id: Date.now().toString(),
      name: `Sesión ${format(new Date(), 'dd/MM/yyyy HH:mm')}`,
      startTime: new Date(),
      duration: 0,
      actions: [],
    };
    setCurrentSession(session);
    setIsRecording(true);
    setActions([]);
    toast.success('Grabación iniciada');
  };

  const handleStopRecording = () => {
    if (currentSession) {
      const endTime = new Date();
      const duration = (endTime.getTime() - currentSession.startTime.getTime()) / 1000;
      const finalSession: Session = {
        ...currentSession,
        endTime,
        duration,
        actions,
      };
      setSessions([finalSession, ...sessions]);
      setCurrentSession(null);
      setIsRecording(false);
      setActions([]);
      toast.success('Grabación detenida');
    }
  };

  const handlePlaySession = (session: Session) => {
    toast.info(`Reproduciendo sesión: ${session.name}`);
    // Would replay session actions
  };

  const handleDeleteSession = (id: string) => {
    setSessions(sessions.filter((s) => s.id !== id));
    toast.success('Sesión eliminada');
  };

  const handleExportSession = (session: Session) => {
    const dataStr = JSON.stringify(session, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${session.name}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Sesión exportada');
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Video className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Grabador de Sesiones</h3>
          </div>
          <div className="flex gap-2">
            {!isRecording ? (
              <button
                onClick={handleStartRecording}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Video className="w-4 h-4" />
                Iniciar Grabación
              </button>
            ) : (
              <button
                onClick={handleStopRecording}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Square className="w-4 h-4" />
                Detener Grabación
              </button>
            )}
          </div>
        </div>

        {/* Recording Status */}
        {isRecording && currentSession && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
              <span className="text-red-400 font-semibold">Grabando...</span>
            </div>
            <p className="text-sm text-gray-300">Sesión: {currentSession.name}</p>
            <p className="text-sm text-gray-300">
              Iniciada: {format(currentSession.startTime, 'HH:mm:ss')}
            </p>
            <p className="text-sm text-gray-300">Acciones: {actions.length}</p>
          </div>
        )}

        {/* Sessions List */}
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-3">
            Sesiones Guardadas ({sessions.length})
          </h4>
          {sessions.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <Video className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No hay sesiones grabadas</p>
            </div>
          ) : (
            <div className="space-y-3">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <h4 className="font-semibold text-white mb-1">{session.name}</h4>
                      <div className="flex gap-4 text-sm text-gray-300">
                        <span>Duración: {formatDuration(session.duration)}</span>
                        <span>Acciones: {session.actions.length}</span>
                      </div>
                      <p className="text-xs text-gray-400 mt-2">
                        {format(session.startTime, 'dd/MM/yyyy HH:mm')}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handlePlaySession(session)}
                        className="p-2 bg-primary-600 hover:bg-primary-700 text-white rounded transition-colors"
                        title="Reproducir"
                      >
                        <Play className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleExportSession(session)}
                        className="p-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
                        title="Exportar"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDeleteSession(session.id)}
                        className="p-2 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                        title="Eliminar"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


