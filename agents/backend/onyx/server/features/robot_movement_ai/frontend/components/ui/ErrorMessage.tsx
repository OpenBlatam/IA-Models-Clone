'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { XCircle, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface ErrorMessageProps {
  title?: string;
  message: string;
  onClose?: () => void;
  className?: string;
  icon?: ReactNode;
  dismissible?: boolean;
}

export default function ErrorMessage({
  title,
  message,
  onClose,
  className,
  icon,
  dismissible = true,
}: ErrorMessageProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={cn(
        'flex items-start gap-tesla-sm p-tesla-md rounded-lg border',
        'bg-[#fee2e2] border-[#ef4444] text-[#991b1b]',
        className
      )}
      role="alert"
      aria-live="assertive"
    >
      <div className="flex-shrink-0 text-[#ef4444]">
        {icon || <XCircle className="w-5 h-5" />}
      </div>
      <div className="flex-1 min-w-0">
        {title && (
          <h4 className="text-sm font-semibold mb-1" style={{ color: '#991b1b' }}>
            {title}
          </h4>
        )}
        <p className="text-sm" style={{ color: '#991b1b' }}>
          {message}
        </p>
      </div>
      {dismissible && onClose && (
        <button
          onClick={onClose}
          className="flex-shrink-0 text-[#991b1b] hover:opacity-70 transition-opacity p-1 rounded hover:bg-[#ef4444]/10"
          aria-label="Cerrar mensaje"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </motion.div>
  );
}

