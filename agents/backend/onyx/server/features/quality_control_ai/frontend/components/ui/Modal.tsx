'use client';

import { memo, type ReactNode } from 'react';
import { X } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './Dialog';
import { Button } from './Button';

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  showCloseButton?: boolean;
}

const Modal = memo(
  ({
    open,
    onClose,
    title,
    children,
    size = 'md',
    showCloseButton = true,
  }: ModalProps): JSX.Element => {
    const sizeClasses = {
      sm: 'max-w-sm',
      md: 'max-w-md',
      lg: 'max-w-lg',
      xl: 'max-w-xl',
    };

    return (
      <Dialog open={open} onOpenChange={onClose}>
        <DialogContent className={sizeClasses[size]}>
          {(title || showCloseButton) && (
            <DialogHeader>
              {title && <DialogTitle>{title}</DialogTitle>}
              {showCloseButton && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onClose}
                  className="absolute right-4 top-4"
                  aria-label="Close modal"
                  tabIndex={0}
                >
                  <X className="w-4 h-4" aria-hidden="true" />
                </Button>
              )}
            </DialogHeader>
          )}
          {children}
        </DialogContent>
      </Dialog>
    );
  }
);

Modal.displayName = 'Modal';

export default Modal;

