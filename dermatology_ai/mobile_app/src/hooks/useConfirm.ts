import { useState, useCallback } from 'react';

interface ConfirmOptions {
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  type?: 'danger' | 'warning' | 'info';
}

interface ConfirmState extends ConfirmOptions {
  visible: boolean;
  onConfirm?: () => void;
  onCancel?: () => void;
}

export const useConfirm = () => {
  const [confirmState, setConfirmState] = useState<ConfirmState>({
    visible: false,
    title: '',
    message: '',
  });

  const confirm = useCallback(
    (options: ConfirmOptions): Promise<boolean> => {
      return new Promise((resolve) => {
        setConfirmState({
          ...options,
          visible: true,
          onConfirm: () => {
            setConfirmState((prev) => ({ ...prev, visible: false }));
            resolve(true);
          },
          onCancel: () => {
            setConfirmState((prev) => ({ ...prev, visible: false }));
            resolve(false);
          },
        });
      });
    },
    []
  );

  const close = useCallback(() => {
    setConfirmState((prev) => ({ ...prev, visible: false }));
  }, []);

  return {
    confirm,
    close,
    confirmState,
  };
};

