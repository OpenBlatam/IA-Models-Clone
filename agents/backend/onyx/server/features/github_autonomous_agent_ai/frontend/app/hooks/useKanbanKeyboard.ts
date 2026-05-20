import { useEffect } from 'react';

interface UseKanbanKeyboardProps {
  onFocusSearch?: () => void;
  onCreateTask?: () => void;
  onClosePanel?: () => void;
  isPanelOpen?: boolean;
}

export function useKanbanKeyboard({
  onFocusSearch,
  onCreateTask,
  onClosePanel,
  isPanelOpen = false,
}: UseKanbanKeyboardProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Focus search con "/"
      if (e.key === '/' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const target = e.target as HTMLElement;
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
          e.preventDefault();
          if (onFocusSearch) {
            onFocusSearch();
          } else {
            const searchInput = document.querySelector(
              'input[type="text"][placeholder*="Buscar"]'
            ) as HTMLInputElement;
            if (searchInput) {
              searchInput.focus();
            }
          }
        }
      }

      // Crear tarea con "c"
      if (e.key === 'c' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const target = e.target as HTMLElement;
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
          e.preventDefault();
          if (onCreateTask) {
            onCreateTask();
          } else {
            window.location.href = '/agent-control';
          }
        }
      }

      // Cerrar panel con "Escape"
      if (e.key === 'Escape' && isPanelOpen && onClosePanel) {
        onClosePanel();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onFocusSearch, onCreateTask, onClosePanel, isPanelOpen]);
}

