'use client';

import { useEffect, useRef } from 'react';
import { useAppStore } from '@/store/app-store';

export function useNotificationSounds() {
  const { notifications } = useAppStore();
  const lastNotificationId = useRef<string | null>(null);

  useEffect(() => {
    const enabled = localStorage.getItem('bul_notification_sounds') === 'true';
    if (!enabled) return;

    const lastNotification = notifications[0];
    if (!lastNotification || lastNotification.id === lastNotificationId.current) return;

    lastNotificationId.current = lastNotification.id;

    // Create simple beep sound using Web Audio API
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      if (lastNotification.type === 'success') {
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
      } else if (lastNotification.type === 'error') {
        oscillator.frequency.value = 400;
        oscillator.type = 'sawtooth';
      } else {
        oscillator.frequency.value = 600;
        oscillator.type = 'sine';
      }

      gainNode.gain.value = 0.1;
      oscillator.start();
      oscillator.stop(audioContext.currentTime + 0.1);
    } catch (error) {
      // Fallback: silent if Web Audio API not available
      console.debug('Audio notification not available');
    }
  }, [notifications]);
}

export default function NotificationSounds() {
  useNotificationSounds();
  return null;
}

