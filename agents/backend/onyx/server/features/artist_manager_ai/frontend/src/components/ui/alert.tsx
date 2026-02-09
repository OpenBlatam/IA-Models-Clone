'use client';

import { AlertCircle, CheckCircle, Info, AlertTriangle, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AlertProps {
  variant?: 'success' | 'error' | 'warning' | 'info';
  title?: string;
  message: string;
  onClose?: () => void;
  className?: string;
}

const variantStyles = {
  success: {
    container: 'bg-green-50 border-green-200 text-green-800',
    icon: CheckCircle,
    iconColor: 'text-green-600',
  },
  error: {
    container: 'bg-red-50 border-red-200 text-red-800',
    icon: AlertCircle,
    iconColor: 'text-red-600',
  },
  warning: {
    container: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    icon: AlertTriangle,
    iconColor: 'text-yellow-600',
  },
  info: {
    container: 'bg-blue-50 border-blue-200 text-blue-800',
    icon: Info,
    iconColor: 'text-blue-600',
  },
};

const Alert = ({ variant = 'info', title, message, onClose, className }: AlertProps) => {
  const styles = variantStyles[variant];
  const Icon = styles.icon;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (onClose && (e.key === 'Enter' || e.key === ' ')) {
      onClose();
    }
  };

  return (
    <div
      className={cn(
        'border rounded-lg p-4 flex items-start',
        styles.container,
        className
      )}
      role="alert"
    >
      <Icon className={cn('w-5 h-5 mr-3 flex-shrink-0 mt-0.5', styles.iconColor)} />
      <div className="flex-1">
        {title && <h4 className="font-medium mb-1">{title}</h4>}
        <p className="text-sm">{message}</p>
      </div>
      {onClose && (
        <button
          onClick={onClose}
          onKeyDown={handleKeyDown}
          className="ml-3 flex-shrink-0 hover:opacity-70 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-500 rounded p-1"
          aria-label="Cerrar alerta"
          tabIndex={0}
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

export { Alert };

