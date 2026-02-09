'use client';

import { useEffect } from 'react';
import { useKeyboardShortcuts } from '@/lib/utils/keyboard';

export default function KeyboardNavigation() {
  const { registerShortcut } = useKeyboardShortcuts();

  useEffect(() => {
    // Register global navigation shortcuts
    registerShortcut('ctrl+k', () => {
      // Focus search bar
      const searchInput = document.querySelector('[data-search-input]') as HTMLInputElement;
      if (searchInput) {
        searchInput.focus();
      }
    });

    registerShortcut('escape', () => {
      // Close modals
      const modals = document.querySelectorAll('[data-modal]');
      modals.forEach((modal) => {
        (modal as HTMLElement).style.display = 'none';
      });
    });

    registerShortcut('ctrl+/', () => {
      // Show shortcuts guide
      const shortcutsTab = document.querySelector('[data-tab="shortcuts"]') as HTMLElement;
      if (shortcutsTab) {
        shortcutsTab.click();
      }
    });
  }, [registerShortcut]);

  return null; // This component doesn't render anything
}


