'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { Info, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface InfoMessageProps {
  title?: string;
  message: string;
  onClose?: () => void;
  className?: string;
  icon?: ReactNode;
  dismissible?: boolean;
}

export default function InfoMessage({
  title,
  message,
  onClose,
  className,
  icon,
  dismissible = true,
}: InfoMessageProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={cn(
        'flex items-start gap-tesla-sm p-tesla-md rounded-lg border',
        'bg-[#dbeafe] border-[#3b82f6] text-[#1e40af]',
        className
      )}
      role="alert"
      aria-live="polite"
    >
      <div className="flex-shrink-0 text-[#3b82f6]">
        {icon || <Info className="w-5 h-5" />}
      </div>
      <div className="flex-1 min-w-0">
        {title && (
          <h4 className="text-sm font-semibold mb-1" style={{ color: '#1e40af' }}>
            {title}
          </h4>
        )}
        <p className="text-sm" style={{ color: '#1e40af' }}>
          {message}
        </p>
      </div>
      {dismissible && onClose && (
        <button
          onClick={onClose}
          className="flex-shrink-0 text-[#1e40af] hover:opacity-70 transition-opacity p-1 rounded hover:bg-[#3b82f6]/10"
          aria-label="Cerrar mensaje"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </motion.div>
  );
}

