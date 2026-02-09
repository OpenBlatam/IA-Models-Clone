'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Command } from 'lucide-react';

interface KeyboardShortcutsProps {
  onSearch?: () => void;
  onAnalyze?: () => void;
  onCompare?: () => void;
}

export function KeyboardShortcuts({ onSearch, onAnalyze, onCompare }: KeyboardShortcutsProps) {
  const router = useRouter();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K para búsqueda
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        onSearch?.();
      }

      // Ctrl/Cmd + Enter para analizar
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        onAnalyze?.();
      }

      // Ctrl/Cmd + Shift + C para comparar
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        onCompare?.();
      }

      // Escape para cerrar modales
      if (e.key === 'Escape') {
        // Cerrar cualquier modal abierto
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onSearch, onAnalyze, onCompare]);

  return null;
}

export function ShortcutsHelp() {
  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2 mb-3">
        <Command className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Atajos de Teclado</h3>
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Búsqueda</span>
          <kbd className="px-2 py-1 bg-white/10 rounded text-white">Ctrl + K</kbd>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Analizar</span>
          <kbd className="px-2 py-1 bg-white/10 rounded text-white">Ctrl + Enter</kbd>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Comparar</span>
          <kbd className="px-2 py-1 bg-white/10 rounded text-white">Ctrl + Shift + C</kbd>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-300">Cerrar</span>
          <kbd className="px-2 py-1 bg-white/10 rounded text-white">Esc</kbd>
        </div>
      </div>
    </div>
  );
}

