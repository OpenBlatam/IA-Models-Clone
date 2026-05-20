/**
 * Alert component.
 * Reusable alert component with variants.
 */

import { type HTMLAttributes, type ReactNode } from 'react';
import { AlertCircle, CheckCircle, Info, X, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

export type AlertVariant = 'info' | 'success' | 'warning' | 'error';

export interface AlertProps extends HTMLAttributes<HTMLDivElement> {
  variant?: AlertVariant;
  title?: string;
  children: ReactNode;
  dismissible?: boolean;
  onDismiss?: () => void;
}

const variantConfig = {
  info: {
    icon: Info,
    bg: 'bg-blue-900/20',
    border: 'border-blue-500/50',
    text: 'text-blue-300',
    iconColor: 'text-blue-400',
  },
  success: {
    icon: CheckCircle,
    bg: 'bg-green-900/20',
    border: 'border-green-500/50',
    text: 'text-green-300',
    iconColor: 'text-green-400',
  },
  warning: {
    icon: AlertTriangle,
    bg: 'bg-yellow-900/20',
    border: 'border-yellow-500/50',
    text: 'text-yellow-300',
    iconColor: 'text-yellow-400',
  },
  error: {
    icon: AlertCircle,
    bg: 'bg-red-900/20',
    border: 'border-red-500/50',
    text: 'text-red-300',
    iconColor: 'text-red-400',
  },
} as const;

/**
 * Alert component.
 * Provides consistent alert styling with variants.
 *
 * @param props - Component props
 * @returns Alert component
 */
export function Alert({
  variant = 'info',
  title,
  children,
  dismissible = false,
  onDismiss,
  className,
  ...props
}: AlertProps) {
  const config = variantConfig[variant];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        'relative p-4 rounded-lg border',
        config.bg,
        config.border,
        className
      )}
      role="alert"
      {...props}
    >
      <div className="flex items-start gap-3">
        <Icon className={cn('w-5 h-5 flex-shrink-0 mt-0.5', config.iconColor)} aria-hidden="true" />
        <div className="flex-1">
          {title && (
            <h4 className={cn('font-semibold mb-1', config.text)}>{title}</h4>
          )}
          <div className={cn('text-sm', config.text)}>{children}</div>
        </div>
        {dismissible && (
          <button
            onClick={onDismiss}
            className={cn(
              'ml-auto flex-shrink-0',
              'hover:opacity-70 transition-opacity',
              'focus:outline-none focus:ring-2 focus:ring-offset-2 rounded',
              config.iconColor
            )}
            aria-label="Cerrar alerta"
          >
            <X className="w-4 h-4" aria-hidden="true" />
          </button>
        )}
      </div>
    </div>
  );
}

