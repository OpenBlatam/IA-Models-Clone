'use client';

import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { FiClock, FiPlay, FiPause, FiSquare } from 'react-icons/fi';
import { format } from 'date-fns';

export default function TimeTracker() {
  const [isRunning, setIsRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [sessions, setSessions] = useState<Array<{ start: Date; end: Date; duration: number }>>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<Date | null>(null);

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(() => {
        if (startTimeRef.current) {
          setElapsed(Math.floor((Date.now() - startTimeRef.current.getTime()) / 1000));
        }
      }, 1000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning]);

  const start = () => {
    startTimeRef.current = new Date();
    setIsRunning(true);
  };

  const pause = () => {
    setIsRunning(false);
  };

  const stop = () => {
    if (startTimeRef.current) {
      const end = new Date();
      const duration = Math.floor((end.getTime() - startTimeRef.current.getTime()) / 1000);
      setSessions((prev) => [
        { start: startTimeRef.current!, end, duration },
        ...prev,
      ].slice(0, 10));
    }
    setIsRunning(false);
    setElapsed(0);
    startTimeRef.current = null;
  };

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <FiClock size={20} className="text-primary-600" />
        <h3 className="font-semibold text-gray-900 dark:text-white">Temporizador</h3>
      </div>

      <div className="text-center mb-6">
        <div className="text-4xl font-mono font-bold text-gray-900 dark:text-white mb-4">
          {formatTime(elapsed)}
        </div>
        <div className="flex items-center justify-center gap-2">
          {!isRunning ? (
            <button onClick={start} className="btn btn-primary">
              <FiPlay size={18} className="mr-2" />
              Iniciar
            </button>
          ) : (
            <>
              <button onClick={pause} className="btn btn-secondary">
                <FiPause size={18} className="mr-2" />
                Pausar
              </button>
              <button onClick={stop} className="btn btn-danger">
                <FiSquare size={18} className="mr-2" />
                Detener
              </button>
            </>
          )}
        </div>
      </div>

      {sessions.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Sesiones Recientes
          </h4>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {sessions.map((session, index) => (
              <div
                key={index}
                className="text-xs text-gray-600 dark:text-gray-400 p-2 bg-gray-50 dark:bg-gray-700 rounded"
              >
                <div className="flex items-center justify-between">
                  <span>{format(session.start, 'PPp')}</span>
                  <span className="font-mono">{formatTime(session.duration)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


