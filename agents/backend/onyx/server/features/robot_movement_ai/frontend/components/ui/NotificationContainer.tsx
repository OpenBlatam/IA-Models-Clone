'use client';

import { AnimatePresence } from 'framer-motion';
import Notification from './Notification';

export interface NotificationData {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
}

interface NotificationContainerProps {
  notifications: NotificationData[];
  onClose: (id: string) => void;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center';
}

export default function NotificationContainer({
  notifications,
  onClose,
  position = 'top-right',
}: NotificationContainerProps) {
  return (
    <div className="fixed z-50 pointer-events-none">
      <AnimatePresence mode="popLayout">
        {notifications.map((notification, index) => (
          <div
            key={notification.id}
            style={{
              marginTop: index > 0 ? '0.5rem' : '0',
            }}
          >
            <Notification
              {...notification}
              onClose={onClose}
              position={position}
            />
          </div>
        ))}
      </AnimatePresence>
    </div>
  );
}



