'use client';

import { useEffect } from 'react';
import { useRouter } from '@/i18n/routing';
import { toast } from 'sonner';

interface Shortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  action: () => void;
  description: string;
}

export const useKeyboardShortcuts = (shortcuts: Shortcut[]) => {
  const router = useRouter();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      shortcuts.forEach((shortcut) => {
        const ctrlMatch = shortcut.ctrl ? e.ctrlKey || e.metaKey : !e.ctrlKey && !e.metaKey;
        const shiftMatch = shortcut.shift ? e.shiftKey : !e.shiftKey;
        const altMatch = shortcut.alt ? e.altKey : !e.altKey;
        const keyMatch = e.key.toLowerCase() === shortcut.key.toLowerCase();

        if (ctrlMatch && shiftMatch && altMatch && keyMatch) {
          e.preventDefault();
          shortcut.action();
        }
      });
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
};

export const KeyboardShortcuts = () => {
  const router = useRouter();

  useKeyboardShortcuts([
    {
      key: 'k',
      ctrl: true,
      action: () => {
        const event = new KeyboardEvent('keydown', {
          key: 'k',
          ctrlKey: true,
          bubbles: true,
        });
        document.dispatchEvent(event);
      },
      description: 'Abrir paleta de comandos',
    },
    {
      key: 'n',
      ctrl: true,
      action: () => {
        router.push('/posts');
        toast.info('Navegando a crear post...');
      },
      description: 'Crear nuevo post',
    },
    {
      key: 'd',
      ctrl: true,
      action: () => {
        router.push('/dashboard');
      },
      description: 'Ir al dashboard',
    },
  ]);

  return null;
};



