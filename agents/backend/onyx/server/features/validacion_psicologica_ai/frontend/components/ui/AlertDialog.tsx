/**
 * Alert dialog component
 */

'use client';

import React from 'react';
import { Modal } from './Modal';
import { Button } from './Button';
import { AlertCircle, Info, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react';

export interface AlertDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm?: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  variant?: 'default' | 'success' | 'warning' | 'destructive' | 'info';
  showCancel?: boolean;
}

const variantIcons = {
  default: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  destructive: XCircle,
  info: AlertCircle,
};

const variantColors = {
  default: 'text-primary',
  success: 'text-green-600',
  warning: 'text-yellow-600',
  destructive: 'text-red-600',
  info: 'text-blue-600',
};

export const AlertDialog: React.FC<AlertDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Aceptar',
  cancelText = 'Cancelar',
  variant = 'default',
  showCancel = true,
}) => {
  const Icon = variantIcons[variant];

  const handleConfirm = () => {
    if (onConfirm) {
      onConfirm();
    }
    onClose();
  };

  const handleKeyDown = (event: React.KeyboardEvent, action: () => void) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      action();
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="md">
      <div className="p-6">
        <div className="flex items-start gap-4">
          <div className={`flex-shrink-0 ${variantColors[variant]}`}>
            <Icon className="h-6 w-6" aria-hidden="true" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold mb-2">{title}</h3>
            <p className="text-muted-foreground">{message}</p>
          </div>
        </div>
        <div className="flex items-center justify-end gap-2 mt-6">
          {showCancel && (
            <Button
              variant="outline"
              onClick={onClose}
              onKeyDown={(e) => handleKeyDown(e, onClose)}
              aria-label={cancelText}
              tabIndex={0}
            >
              {cancelText}
            </Button>
          )}
          <Button
            variant={variant === 'destructive' ? 'destructive' : 'primary'}
            onClick={handleConfirm}
            onKeyDown={(e) => handleKeyDown(e, handleConfirm)}
            aria-label={confirmText}
            tabIndex={0}
          >
            {confirmText}
          </Button>
        </div>
      </div>
    </Modal>
  );
};



