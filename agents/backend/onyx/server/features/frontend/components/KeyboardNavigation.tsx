'use client';

import { useEffect } from 'react';
import { useAppStore } from '@/store/app-store';

export default function KeyboardNavigation() {
  const { activeView, setActiveView } = useAppStore();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if user is typing in input/textarea
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Number keys for navigation
      if (e.key >= '1' && e.key <= '6') {
        const views = ['dashboard', 'generate', 'tasks', 'documents', 'stats', 'favorites'];
        const index = parseInt(e.key) - 1;
        if (views[index]) {
          e.preventDefault();
          setActiveView(views[index] as any);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setActiveView]);

  return null;
}


