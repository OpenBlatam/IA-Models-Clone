'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface SuccessMessageProps {
  title?: string;
  message: string;
  onClose?: () => void;
  className?: string;
  icon?: ReactNode;
  dismissible?: boolean;
}

export default function SuccessMessage({
  title,
  message,
  onClose,
  className,
  icon,
  dismissible = true,
}: SuccessMessageProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={cn(
        'flex items-start gap-tesla-sm p-tesla-md rounded-lg border',
        'bg-[#d1fae5] border-[#10b981] text-[#065f46]',
        className
      )}
      role="alert"
      aria-live="polite"
    >
      <div className="flex-shrink-0 text-[#10b981]">
        {icon || <CheckCircle className="w-5 h-5" />}
      </div>
      <div className="flex-1 min-w-0">
        {title && (
          <h4 className="text-sm font-semibold mb-1" style={{ color: '#065f46' }}>
            {title}
          </h4>
        )}
        <p className="text-sm" style={{ color: '#065f46' }}>
          {message}
        </p>
      </div>
      {dismissible && onClose && (
        <button
          onClick={onClose}
          className="flex-shrink-0 text-[#065f46] hover:opacity-70 transition-opacity p-1 rounded hover:bg-[#10b981]/10"
          aria-label="Cerrar mensaje"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </motion.div>
  );
}

