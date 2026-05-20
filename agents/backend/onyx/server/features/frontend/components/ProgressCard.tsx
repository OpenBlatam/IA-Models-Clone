'use client';

import { motion, AnimatePresence } from 'framer-motion';
import type { TaskStatus } from '@/types/api';

interface ProgressCardProps {
  task: TaskStatus;
  onCancel?: () => void;
}

export default function ProgressCard({ task, onCancel }: ProgressCardProps) {
  const getStatusColor = () => {
    switch (task.status) {
      case 'queued':
        return 'bg-blue-500';
      case 'processing':
        return 'bg-yellow-500';
      case 'completed':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    switch (task.status) {
      case 'queued':
        return 'En cola...';
      case 'processing':
        return 'Procesando...';
      case 'completed':
        return 'Completado';
      case 'failed':
        return 'Fallido';
      default:
        return task.status;
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="card border-l-4 border-l-primary-500"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Generando Documento</h3>
            <p className="text-sm text-gray-600 mt-1">{getStatusText()}</p>
          </div>
          {onCancel && (task.status === 'queued' || task.status === 'processing') && (
            <button
              onClick={onCancel}
              className="btn-icon text-red-600 hover:bg-red-50"
              title="Cancelar"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M15 9l-6 6" />
                <path d="M9 9l6 6" />
              </svg>
            </button>
          )}
        </div>

        <div className="space-y-3">
          <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden">
            <motion.div
              className={`h-full ${getStatusColor()}`}
              initial={{ width: 0 }}
              animate={{ width: `${task.progress}%` }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
            />
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">{task.progress}%</span>
            {task.processing_time && (
              <span className="text-gray-500">
                Tiempo: {task.processing_time.toFixed(1)}s
              </span>
            )}
          </div>
        </div>

        {task.error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg"
          >
            <p className="text-sm text-red-800">{task.error}</p>
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}


