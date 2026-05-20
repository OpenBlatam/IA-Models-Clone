import * as React from 'react';
import { X, CheckCircle2, AlertCircle, Info, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ToastProps {
  message: string;
  type?: 'success' | 'error' | 'info' | 'warning';
  onClose?: () => void;
  duration?: number;
}

const Toast = ({ message, type = 'info', onClose, duration = 5000 }: ToastProps) => {
  const [isVisible, setIsVisible] = React.useState(true);

  React.useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        setIsVisible(false);
        setTimeout(() => onClose?.(), 300);
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [duration, onClose]);

  const icons = {
    success: CheckCircle2,
    error: AlertCircle,
    info: Info,
    warning: AlertTriangle,
  };

  const styles = {
    success: 'bg-green-50 border-green-200 text-green-800',
    error: 'bg-red-50 border-red-200 text-red-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
  };

  const Icon = icons[type];

  if (!isVisible) return null;

  return (
    <div
      className={cn(
        'fixed bottom-4 right-4 z-50 flex items-center gap-3 rounded-lg border p-4 shadow-lg transition-all',
        styles[type]
      )}
      role="alert"
      aria-live="assertive"
    >
      <Icon className="h-5 w-5 flex-shrink-0" aria-hidden="true" />
      <p className="text-sm font-medium">{message}</p>
      {onClose && (
        <button
          onClick={onClose}
          className="ml-auto rounded-sm opacity-70 hover:opacity-100 focus:outline-none"
          aria-label="Close toast"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
};

export default Toast;




