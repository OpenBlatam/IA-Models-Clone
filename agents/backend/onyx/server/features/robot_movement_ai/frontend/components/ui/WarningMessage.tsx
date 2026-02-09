'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface WarningMessageProps {
  title?: string;
  message: string;
  onClose?: () => void;
  className?: string;
  icon?: ReactNode;
  dismissible?: boolean;
}

export default function WarningMessage({
  title,
  message,
  onClose,
  className,
  icon,
  dismissible = true,
}: WarningMessageProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={cn(
        'flex items-start gap-tesla-sm p-tesla-md rounded-lg border',
        'bg-[#fef3c7] border-[#f59e0b] text-[#92400e]',
        className
      )}
      role="alert"
      aria-live="polite"
    >
      <div className="flex-shrink-0 text-[#f59e0b]">
        {icon || <AlertTriangle className="w-5 h-5" />}
      </div>
      <div className="flex-1 min-w-0">
        {title && (
          <h4 className="text-sm font-semibold mb-1" style={{ color: '#92400e' }}>
            {title}
          </h4>
        )}
        <p className="text-sm" style={{ color: '#92400e' }}>
          {message}
        </p>
      </div>
      {dismissible && onClose && (
        <button
          onClick={onClose}
          className="flex-shrink-0 text-[#92400e] hover:opacity-70 transition-opacity p-1 rounded hover:bg-[#f59e0b]/10"
          aria-label="Cerrar mensaje"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </motion.div>
  );
}

