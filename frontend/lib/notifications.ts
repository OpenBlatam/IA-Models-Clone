interface Notification {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export function addNotification(notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) {
  const newNotification: Notification = {
    ...notification,
    id: Date.now().toString(),
    timestamp: new Date(),
    read: false,
  };

  const stored = localStorage.getItem('bul_notifications');
  const notifications: Notification[] = stored ? JSON.parse(stored) : [];
  
  const updated = [newNotification, ...notifications].slice(0, 50); // Keep last 50
  localStorage.setItem('bul_notifications', JSON.stringify(updated));

  // Trigger custom event for same tab
  window.dispatchEvent(new CustomEvent('bul_notification_update'));
  
  // Trigger storage event for other tabs
  window.dispatchEvent(new StorageEvent('storage', {
    key: 'bul_notifications',
    newValue: JSON.stringify(updated),
  }));
}

export function createTaskNotification(taskId: string, status: string) {
  if (status === 'completed') {
    addNotification({
      type: 'success',
      title: 'Documento Generado',
      message: `Tu documento ha sido generado exitosamente`,
      action: {
        label: 'Ver documento',
        onClick: () => {
          window.location.href = `/tasks/${taskId}`;
        },
      },
    });
    
    // Add activity
    window.dispatchEvent(new CustomEvent('bul_activity', {
      detail: {
        id: Date.now().toString(),
        type: 'task_completed',
        message: 'Documento generado exitosamente',
        timestamp: new Date(),
        taskId,
      },
    }));
  } else if (status === 'failed') {
    addNotification({
      type: 'error',
      title: 'Error en Generación',
      message: `La generación del documento falló`,
    });
    
    // Add activity
    window.dispatchEvent(new CustomEvent('bul_activity', {
      detail: {
        id: Date.now().toString(),
        type: 'task_failed',
        message: 'Error al generar documento',
        timestamp: new Date(),
        taskId,
      },
    }));
  }
}

