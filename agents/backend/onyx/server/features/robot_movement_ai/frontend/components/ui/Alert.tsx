'use client';

import { HTMLAttributes, ReactNode } from 'react';
import { AlertCircle, CheckCircle, Info, XCircle, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import { motion, AnimatePresence } from 'framer-motion';

interface AlertProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'success' | 'error' | 'warning' | 'info';
  title?: string;
  description?: string;
  icon?: ReactNode;
  onClose?: () => void;
  dismissible?: boolean;
}

const icons = {
  success: CheckCircle,
  error: XCircle,
  warning: AlertCircle,
  info: Info,
};

const variants = {
  success: 'bg-green-50 border-green-200 text-green-800',
  error: 'bg-red-50 border-red-200 text-red-800',
  warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
  info: 'bg-blue-50 border-blue-200 text-blue-800',
};

export default function Alert({
  variant = 'info',
  title,
  description,
  icon,
  onClose,
  dismissible = false,
  className,
  children,
  ...props
}: AlertProps) {
  const Icon = icon || icons[variant];

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className={cn(
          'relative rounded-lg border p-4',
          variants[variant],
          className
        )}
        role="alert"
        {...props}
      >
        <div className="flex items-start gap-3">
          <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            {title && (
              <h4 className="font-semibold mb-1 text-sm">{title}</h4>
            )}
            {description && (
              <p className="text-sm">{description}</p>
            )}
            {children}
          </div>
          {dismissible && onClose && (
            <button
              onClick={onClose}
              className="flex-shrink-0 p-1 rounded-md hover:bg-black/5 transition-colors min-h-[32px] min-w-[32px] flex items-center justify-center"
              aria-label="Cerrar alerta"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}



