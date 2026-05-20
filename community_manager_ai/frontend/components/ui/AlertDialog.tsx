'use client';

import { ReactNode } from 'react';
import * as AlertDialogPrimitive from '@radix-ui/react-alert-dialog';
import { AlertTriangle, Info, CheckCircle } from 'lucide-react';
import { Button } from './Button';
import { cn } from '@/lib/utils';

interface AlertDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  variant?: 'danger' | 'warning' | 'info' | 'success';
  loading?: boolean;
}

const variantConfig = {
  danger: {
    icon: AlertTriangle,
    iconColor: 'text-red-600 dark:text-red-400',
    confirmVariant: 'danger' as const,
  },
  warning: {
    icon: AlertTriangle,
    iconColor: 'text-yellow-600 dark:text-yellow-400',
    confirmVariant: 'secondary' as const,
  },
  info: {
    icon: Info,
    iconColor: 'text-blue-600 dark:text-blue-400',
    confirmVariant: 'primary' as const,
  },
  success: {
    icon: CheckCircle,
    iconColor: 'text-green-600 dark:text-green-400',
    confirmVariant: 'primary' as const,
  },
};

export const AlertDialog = ({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = 'Confirmar',
  cancelLabel = 'Cancelar',
  onConfirm,
  variant = 'danger',
  loading = false,
}: AlertDialogProps) => {
  const config = variantConfig[variant];
  const Icon = config.icon;

  return (
    <AlertDialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialogPrimitive.Portal>
        <AlertDialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm" />
        <AlertDialogPrimitive.Content
          className={cn(
            'fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2',
            'rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6 shadow-xl'
          )}
        >
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className={cn('flex-shrink-0', config.iconColor)}>
                <Icon className="h-6 w-6" />
              </div>
              <div className="flex-1">
                <AlertDialogPrimitive.Title className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {title}
                </AlertDialogPrimitive.Title>
                <AlertDialogPrimitive.Description className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  {description}
                </AlertDialogPrimitive.Description>
              </div>
            </div>
            <div className="flex justify-end gap-3">
              <AlertDialogPrimitive.Cancel asChild>
                <Button variant="ghost" disabled={loading}>
                  {cancelLabel}
                </Button>
              </AlertDialogPrimitive.Cancel>
              <AlertDialogPrimitive.Action asChild>
                <Button
                  variant={config.confirmVariant}
                  onClick={onConfirm}
                  disabled={loading}
                  loading={loading}
                >
                  {confirmLabel}
                </Button>
              </AlertDialogPrimitive.Action>
            </div>
          </div>
        </AlertDialogPrimitive.Content>
      </AlertDialogPrimitive.Portal>
    </AlertDialogPrimitive.Root>
  );
};



