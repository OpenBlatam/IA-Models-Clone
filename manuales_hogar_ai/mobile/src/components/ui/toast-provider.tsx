/**
 * Toast Provider
 * ==============
 * Provider component for global toast notifications
 */

import { useEffect, useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Toast } from './toast';
import { toastManager } from '@/lib/utils/toast-manager';

interface ToastState {
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  visible: boolean;
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toast, setToast] = useState<ToastState>({
    message: '',
    type: 'info',
    visible: false,
  });

  useEffect(() => {
    const handleShow = (options: {
      message: string;
      type?: 'success' | 'error' | 'info' | 'warning';
      duration?: number;
    }) => {
      setToast({
        message: options.message,
        type: options.type || 'info',
        visible: true,
      });

      setTimeout(() => {
        setToast((prev) => ({ ...prev, visible: false }));
      }, options.duration || 3000);
    };

    toastManager.on('show', handleShow);

    return () => {
      toastManager.off('show', handleShow);
    };
  }, []);

  return (
    <>
      {children}
      <SafeAreaView style={styles.toastContainer} edges={['top']} pointerEvents="box-none">
        <Toast
          message={toast.message}
          type={toast.type}
          visible={toast.visible}
          onHide={() => setToast((prev) => ({ ...prev, visible: false }))}
        />
      </SafeAreaView>
    </>
  );
}

const styles = StyleSheet.create({
  toastContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 9999,
  },
});



