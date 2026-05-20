'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';

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
    const handleKeyDown = (event: KeyboardEvent) => {
      shortcuts.forEach((shortcut) => {
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatch = shortcut.ctrl ? event.ctrlKey || event.metaKey : !event.ctrlKey && !event.metaKey;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const altMatch = shortcut.alt ? event.altKey : !event.altKey;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
          event.preventDefault();
          shortcut.action();
        }
      });
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
};

export const useDefaultShortcuts = () => {
  const router = useRouter();

  useKeyboardShortcuts([
    {
      key: 'k',
      ctrl: true,
      action: () => {
        // Command palette will be opened by CommandPaletteProvider
        const event = new KeyboardEvent('keydown', {
          key: 'k',
          ctrlKey: true,
          metaKey: true,
        });
        window.dispatchEvent(event);
      },
      description: 'Abrir paleta de comandos',
    },
    {
      key: 'h',
      ctrl: true,
      action: () => router.push('/'),
      description: 'Ir al inicio',
    },
    {
      key: 'd',
      ctrl: true,
      action: () => router.push('/dashboard'),
      description: 'Ir al dashboard',
    },
    {
      key: 'p',
      ctrl: true,
      action: () => router.push('/products'),
      description: 'Ir a productos',
    },
  ]);
};

