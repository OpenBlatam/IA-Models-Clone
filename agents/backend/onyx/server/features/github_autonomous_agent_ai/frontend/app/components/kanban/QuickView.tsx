'use client';

import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { Task } from '../../types/task';
import { cn } from '../../utils/cn';

interface QuickViewProps {
  task: Task | null;
}

export function QuickView({ task }: QuickViewProps) {
  if (!task) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 10 }}
      className="fixed z-50 bg-white border border-gray-200 rounded-lg shadow-xl p-4 max-w-md pointer-events-none"
      style={{
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
      }}
    >
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-600">{task.repository}</span>
          <span className={cn(
            "px-2 py-0.5 rounded-full text-xs font-medium",
            task.status === 'completed' ? "bg-green-100 text-green-800" :
            task.status === 'failed' ? "bg-red-100 text-red-800" :
            task.status === 'processing' || task.status === 'running' ? "bg-blue-100 text-blue-800" :
            "bg-gray-100 text-gray-800"
          )}>
            {task.status}
          </span>
        </div>
        <p className="text-sm font-medium text-gray-900">{task.instruction}</p>
        {task.error && (
          <p className="text-xs text-red-600 line-clamp-2">⚠️ {task.error}</p>
        )}
        {task.executionResult?.success && (
          <p className="text-xs text-green-600">✓ Commit exitoso</p>
        )}
        <div className="text-xs text-gray-400">
          {format(new Date(task.createdAt), 'dd MMM yyyy, HH:mm', { locale: es })}
        </div>
      </div>
    </motion.div>
  );
}

