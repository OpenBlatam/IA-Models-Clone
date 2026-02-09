import { useEffect } from 'react';
import { cn } from '@/lib/utils';

type ToastVariant = 'success' | 'error' | 'warning' | 'info';

interface ToastProps {
  message: string;
  variant?: ToastVariant;
  isVisible: boolean;
  onClose: () => void;
  duration?: number;
}

const VARIANT_CLASSES: Record<ToastVariant, string> = {
  success: 'bg-green-50 text-green-800 border-green-200',
  error: 'bg-red-50 text-red-800 border-red-200',
  warning: 'bg-yellow-50 text-yellow-800 border-yellow-200',
  info: 'bg-blue-50 text-blue-800 border-blue-200',
};

const DEFAULT_DURATION = 5000;

const Toast = ({
  message,
  variant = 'info',
  isVisible,
  onClose,
  duration = DEFAULT_DURATION,
}: ToastProps): JSX.Element => {
  useEffect(() => {
    if (!isVisible) {
      return;
    }

    const timer = setTimeout(() => {
      onClose();
    }, duration);

    return () => {
      clearTimeout(timer);
    };
  }, [isVisible, duration, onClose]);

  if (!isVisible) {
    return <></>;
  }

  const variantClass = VARIANT_CLASSES[variant];

  return (
    <div
      className={cn(
        'fixed top-4 right-4 p-4 rounded-lg border shadow-lg z-50 min-w-[300px]',
        variantClass
      )}
      role="alert"
      aria-live="assertive"
    >
      <div className="flex items-center justify-between">
        <p className="font-medium">{message}</p>
        <button
          onClick={onClose}
          className="ml-4 text-current opacity-70 hover:opacity-100 transition-opacity"
          aria-label="Close toast"
          tabIndex={0}
        >
          <span className="text-xl">&times;</span>
        </button>
      </div>
    </div>
  );
};

export default Toast;



