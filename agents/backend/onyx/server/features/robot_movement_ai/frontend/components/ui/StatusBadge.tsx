'use client';

import { motion } from 'framer-motion';
import { CheckCircle, XCircle, AlertCircle, Clock, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

type Status = 'success' | 'error' | 'warning' | 'info' | 'loading' | 'pending';

interface StatusBadgeProps {
  status: Status;
  label: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'outline' | 'solid';
  className?: string;
  showIcon?: boolean;
}

export default function StatusBadge({
  status,
  label,
  size = 'md',
  variant = 'default',
  className,
  showIcon = true,
}: StatusBadgeProps) {
  const icons = {
    success: CheckCircle,
    error: XCircle,
    warning: AlertCircle,
    info: AlertCircle,
    loading: Loader2,
    pending: Clock,
  };

  const colors = {
    success: {
      default: 'bg-green-50 text-green-700 border-green-200',
      outline: 'border-green-500 text-green-700 bg-transparent',
      solid: 'bg-green-600 text-white border-green-600',
    },
    error: {
      default: 'bg-red-50 text-red-700 border-red-200',
      outline: 'border-red-500 text-red-700 bg-transparent',
      solid: 'bg-red-600 text-white border-red-600',
    },
    warning: {
      default: 'bg-yellow-50 text-yellow-700 border-yellow-200',
      outline: 'border-yellow-500 text-yellow-700 bg-transparent',
      solid: 'bg-yellow-600 text-white border-yellow-600',
    },
    info: {
      default: 'bg-blue-50 text-blue-700 border-blue-200',
      outline: 'border-blue-500 text-blue-700 bg-transparent',
      solid: 'bg-tesla-blue text-white border-tesla-blue',
    },
    loading: {
      default: 'bg-gray-50 text-tesla-gray-dark border-gray-200',
      outline: 'border-gray-500 text-tesla-gray-dark bg-transparent',
      solid: 'bg-gray-600 text-white border-gray-600',
    },
    pending: {
      default: 'bg-gray-50 text-tesla-gray-dark border-gray-200',
      outline: 'border-gray-500 text-tesla-gray-dark bg-transparent',
      solid: 'bg-gray-600 text-white border-gray-600',
    },
  };

  const sizes = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1.5 text-sm',
    lg: 'px-4 py-2 text-base',
  };

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
  };

  const Icon = icons[status];
  const colorClasses = colors[status][variant];

  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        'inline-flex items-center gap-2 rounded-md border font-medium transition-all',
        sizes[size],
        colorClasses,
        className
      )}
    >
      {showIcon && Icon && (
        <Icon
          className={cn(
            iconSizes[size],
            status === 'loading' && 'animate-spin'
          )}
        />
      )}
      {label}
    </motion.span>
  );
}



