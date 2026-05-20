import { useState, useCallback } from 'react';

interface ConfirmDialogOptions {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'danger' | 'warning' | 'info';
}

export const useConfirmDialog = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [options, setOptions] = useState<ConfirmDialogOptions>({
    title: '',
    message: '',
  });
  const [onConfirmCallback, setOnConfirmCallback] = useState<(() => void) | null>(null);

  const openDialog = useCallback((opts: ConfirmDialogOptions, onConfirm: () => void) => {
    setOptions(opts);
    setOnConfirmCallback(() => onConfirm);
    setIsOpen(true);
  }, []);

  const closeDialog = useCallback(() => {
    setIsOpen(false);
    setOnConfirmCallback(null);
  }, []);

  const handleConfirm = useCallback(() => {
    if (onConfirmCallback) {
      onConfirmCallback();
    }
    closeDialog();
  }, [onConfirmCallback, closeDialog]);

  return {
    isOpen,
    options,
    openDialog,
    closeDialog,
    handleConfirm,
  };
};

