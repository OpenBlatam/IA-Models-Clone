import React, { createContext, useContext, useState, useCallback } from 'react';
import { Snackbar } from '../components/common/snackbar';

interface SnackbarState {
  message: string;
  visible: boolean;
  type: 'default' | 'success' | 'error' | 'warning' | 'info';
  actionLabel?: string;
  onAction?: () => void;
  duration?: number;
}

interface SnackbarContextValue {
  showSnackbar: (
    message: string,
    type?: 'default' | 'success' | 'error' | 'warning' | 'info',
    options?: {
      duration?: number;
      actionLabel?: string;
      onAction?: () => void;
    }
  ) => void;
}

const SnackbarContext = createContext<SnackbarContextValue | undefined>(undefined);

export function SnackbarProvider({ children }: { children: React.ReactNode }) {
  const [snackbar, setSnackbar] = useState<SnackbarState>({
    message: '',
    visible: false,
    type: 'default',
  });

  const showSnackbar = useCallback(
    (
      message: string,
      type: 'default' | 'success' | 'error' | 'warning' | 'info' = 'default',
      options?: {
        duration?: number;
        actionLabel?: string;
        onAction?: () => void;
      }
    ) => {
      setSnackbar({
        message,
        type,
        visible: true,
        duration: options?.duration,
        actionLabel: options?.actionLabel,
        onAction: options?.onAction,
      });
    },
    []
  );

  const hideSnackbar = useCallback(() => {
    setSnackbar((prev) => ({ ...prev, visible: false }));
  }, []);

  return (
    <SnackbarContext.Provider value={{ showSnackbar }}>
      {children}
      <Snackbar
        message={snackbar.message}
        visible={snackbar.visible}
        type={snackbar.type}
        duration={snackbar.duration}
        actionLabel={snackbar.actionLabel}
        onAction={snackbar.onAction}
        onDismiss={hideSnackbar}
      />
    </SnackbarContext.Provider>
  );
}

export function useSnackbar() {
  const context = useContext(SnackbarContext);
  if (context === undefined) {
    throw new Error('useSnackbar must be used within a SnackbarProvider');
  }
  return context;
}

