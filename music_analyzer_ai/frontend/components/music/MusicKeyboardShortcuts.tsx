'use client';

import { useState, useEffect } from 'react';
import { Keyboard, X } from 'lucide-react';

interface Shortcut {
  key: string;
  description: string;
  category: string;
}

export function MusicKeyboardShortcuts() {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const shortcuts: Shortcut[] = [
    { key: 'Ctrl/Cmd + K', description: 'Buscar', category: 'Navegación' },
    { key: 'Space', description: 'Play/Pause', category: 'Reproductor' },
    { key: '→', description: 'Siguiente canción', category: 'Reproductor' },
    { key: '←', description: 'Canción anterior', category: 'Reproductor' },
    { key: '↑', description: 'Subir volumen', category: 'Reproductor' },
    { key: '↓', description: 'Bajar volumen', category: 'Reproductor' },
    { key: 'M', description: 'Mute/Unmute', category: 'Reproductor' },
    { key: 'S', description: 'Shuffle', category: 'Reproductor' },
    { key: 'R', description: 'Repeat', category: 'Reproductor' },
    { key: '1', description: 'Tab: Buscar', category: 'Navegación' },
    { key: '2', description: 'Tab: Análisis', category: 'Navegación' },
    { key: '3', description: 'Tab: Comparar', category: 'Navegación' },
    { key: '4', description: 'Tab: Recomendaciones', category: 'Navegación' },
    { key: 'Esc', description: 'Cerrar modales', category: 'Navegación' },
    { key: '?', description: 'Mostrar/Ocultar atajos', category: 'Navegación' },
  ];

  const filteredShortcuts = shortcuts.filter(
    (shortcut) =>
      shortcut.key.toLowerCase().includes(searchQuery.toLowerCase()) ||
      shortcut.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      shortcut.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const groupedShortcuts = filteredShortcuts.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) {
      acc[shortcut.category] = [];
    }
    acc[shortcut.category].push(shortcut);
    return acc;
  }, {} as Record<string, Shortcut[]>);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === '?' && (e.ctrlKey || e.metaKey)) {
        setIsOpen(!isOpen);
      }
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isOpen]);

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="px-3 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors flex items-center gap-2"
        title="Atajos de teclado (Ctrl/Cmd + ?)"
      >
        <Keyboard className="w-4 h-4" />
        <span className="text-sm">Atajos</span>
      </button>
    );
  }

  return (
    <>
      <div
        className="fixed inset-0 bg-black/50 z-50"
        onClick={() => setIsOpen(false)}
      />
      <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
        <div className="bg-gray-800 rounded-xl shadow-2xl border border-white/20 max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
          <div className="flex items-center justify-between p-6 border-b border-white/10">
            <div className="flex items-center gap-2">
              <Keyboard className="w-5 h-5 text-purple-300" />
              <h2 className="text-2xl font-semibold text-white">Atajos de Teclado</h2>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="p-2 text-gray-400 hover:text-white rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="p-6 overflow-y-auto flex-1">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Buscar atajo..."
              className="w-full px-4 py-2 mb-6 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-400"
            />

            <div className="space-y-6">
              {Object.entries(groupedShortcuts).map(([category, items]) => (
                <div key={category}>
                  <h3 className="text-white font-semibold mb-3">{category}</h3>
                  <div className="space-y-2">
                    {items.map((shortcut, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10"
                      >
                        <span className="text-gray-300">{shortcut.description}</span>
                        <kbd className="px-3 py-1 bg-white/10 border border-white/20 rounded text-sm text-white font-mono">
                          {shortcut.key}
                        </kbd>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}


