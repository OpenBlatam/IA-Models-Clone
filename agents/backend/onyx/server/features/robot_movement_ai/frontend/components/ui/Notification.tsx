'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle, XCircle, AlertCircle, Info, Bell } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import { useEffect } from 'react';

type NotificationType = 'success' | 'error' | 'warning' | 'info';

interface NotificationProps {
  id: string;
  type: NotificationType;
  title: string;
  message?: string;
  duration?: number;
  onClose: (id: string) => void;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center';
}

export default function Notification({
  id,
  type,
  title,
  message,
  duration = 5000,
  onClose,
  position = 'top-right',
}: NotificationProps) {
  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        onClose(id);
      }, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, id, onClose]);

  const icons = {
    success: CheckCircle,
    error: XCircle,
    warning: AlertCircle,
    info: Info,
  };

  const colors = {
    success: {
      bg: 'bg-green-50',
      border: 'border-green-200',
      icon: 'text-green-600',
      title: 'text-green-900',
      message: 'text-green-700',
    },
    error: {
      bg: 'bg-red-50',
      border: 'border-red-200',
      icon: 'text-red-600',
      title: 'text-red-900',
      message: 'text-red-700',
    },
    warning: {
      bg: 'bg-yellow-50',
      border: 'border-yellow-200',
      icon: 'text-yellow-600',
      title: 'text-yellow-900',
      message: 'text-yellow-700',
    },
    info: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      icon: 'text-tesla-blue',
      title: 'text-blue-900',
      message: 'text-blue-700',
    },
  };

  const positions = {
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'top-center': 'top-4 left-1/2 -translate-x-1/2',
    'bottom-center': 'bottom-4 left-1/2 -translate-x-1/2',
  };

  const Icon = icons[type];
  const colorClasses = colors[type];

  return (
    <motion.div
      initial={{ opacity: 0, y: -20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.2 } }}
      className={cn(
        'fixed z-50 w-full max-w-sm',
        positions[position]
      )}
    >
      <div
        className={cn(
          'rounded-lg border shadow-tesla-lg p-4',
          colorClasses.bg,
          colorClasses.border
        )}
      >
        <div className="flex items-start gap-3">
          <Icon className={cn('w-5 h-5 flex-shrink-0 mt-0.5', colorClasses.icon)} />
          <div className="flex-1 min-w-0">
            <p className={cn('text-sm font-semibold', colorClasses.title)}>
              {title}
            </p>
            {message && (
              <p className={cn('text-sm mt-1', colorClasses.message)}>
                {message}
              </p>
            )}
          </div>
          <button
            onClick={() => onClose(id)}
            className={cn(
              'flex-shrink-0 p-1 rounded-md hover:bg-black/5 transition-colors min-h-[32px] min-w-[32px] flex items-center justify-center',
              colorClasses.icon
            )}
            aria-label="Cerrar notificación"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </motion.div>
  );
}



