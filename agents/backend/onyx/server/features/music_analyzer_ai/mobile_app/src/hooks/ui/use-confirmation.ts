import { useState, useCallback } from 'react';

export interface ConfirmationOptions {
  title?: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm?: () => void | Promise<void>;
  onCancel?: () => void;
}

export function useConfirmation() {
  const [isOpen, setIsOpen] = useState(false);
  const [options, setOptions] = useState<ConfirmationOptions | null>(null);

  const confirm = useCallback(
    async (opts: ConfirmationOptions) => {
      setOptions(opts);
      setIsOpen(true);
    },
    []
  );

  const handleConfirm = useCallback(async () => {
    if (options?.onConfirm) {
      await options.onConfirm();
    }
    setIsOpen(false);
    setOptions(null);
  }, [options]);

  const handleCancel = useCallback(() => {
    if (options?.onCancel) {
      options.onCancel();
    }
    setIsOpen(false);
    setOptions(null);
  }, [options]);

  return {
    isOpen,
    options,
    confirm,
    handleConfirm,
    handleCancel,
  };
}


