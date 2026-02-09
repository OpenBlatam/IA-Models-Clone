'use client';

import { X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChipProps {
  label: string;
  onRemove?: () => void;
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger';
  size?: 'sm' | 'md';
  className?: string;
}

const variantClasses = {
  default: 'bg-gray-100 text-gray-800',
  primary: 'bg-blue-100 text-blue-800',
  success: 'bg-green-100 text-green-800',
  warning: 'bg-yellow-100 text-yellow-800',
  danger: 'bg-red-100 text-red-800',
};

const sizeClasses = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-sm',
};

const Chip = ({ label, onRemove, variant = 'default', size = 'md', className }: ChipProps) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (onRemove && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      onRemove();
    }
  };

  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full font-medium',
        variantClasses[variant],
        sizeClasses[size],
        className
      )}
    >
      {label}
      {onRemove && (
        <button
          onClick={onRemove}
          onKeyDown={handleKeyDown}
          className="ml-1.5 hover:opacity-70 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-500 rounded-full"
          aria-label={`Eliminar ${label}`}
          tabIndex={0}
        >
          <X className="w-3 h-3" />
        </button>
      )}
    </span>
  );
};

export { Chip };

