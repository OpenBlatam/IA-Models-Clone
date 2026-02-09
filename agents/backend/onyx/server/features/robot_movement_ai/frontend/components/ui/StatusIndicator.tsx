'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface StatusIndicatorProps {
  status: 'online' | 'offline' | 'away' | 'busy' | 'loading';
  size?: 'sm' | 'md' | 'lg';
  showPulse?: boolean;
  className?: string;
  label?: string;
}

const sizeClasses = {
  sm: 'w-2 h-2',
  md: 'w-3 h-3',
  lg: 'w-4 h-4',
};

const statusColors = {
  online: 'bg-[#10b981]',
  offline: 'bg-[#9ca3af]',
  away: 'bg-[#f59e0b]',
  busy: 'bg-[#ef4444]',
  loading: 'bg-tesla-blue',
};

const pulseColors = {
  online: 'bg-[#10b981]/30',
  offline: 'bg-[#9ca3af]/30',
  away: 'bg-[#f59e0b]/30',
  busy: 'bg-[#ef4444]/30',
  loading: 'bg-tesla-blue/30',
};

export default function StatusIndicator({
  status,
  size = 'md',
  showPulse = true,
  className,
  label,
}: StatusIndicatorProps) {
  return (
    <div className={cn('flex items-center gap-tesla-sm', className)}>
      <div className="relative">
        <motion.div
          className={cn('rounded-full', sizeClasses[size], statusColors[status])}
          animate={status === 'loading' ? { opacity: [1, 0.5, 1] } : {}}
          transition={
            status === 'loading'
              ? { duration: 1.5, repeat: Infinity, ease: 'easeInOut' }
              : {}
          }
        />
        {showPulse && status === 'online' && (
          <motion.div
            className={cn(
              'absolute inset-0 rounded-full',
              sizeClasses[size],
              pulseColors[status]
            )}
            animate={{ scale: [1, 2, 2], opacity: [0.5, 0, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeOut' }}
          />
        )}
      </div>
      {label && (
        <span className="text-sm text-tesla-gray-dark capitalize">{label || status}</span>
      )}
    </div>
  );
}

