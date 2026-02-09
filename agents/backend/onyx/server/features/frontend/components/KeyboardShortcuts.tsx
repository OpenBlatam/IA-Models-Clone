'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX, FiCommand } from 'react-icons/fi';
import { useHotkeys } from 'react-hotkeys-hook';

interface Shortcut {
  keys: string;
  description: string;
  category: string;
}

const shortcuts: Shortcut[] = [
  {
    keys: 'Ctrl + Enter',
    description: 'Generar documento',
    category: 'Generación',
  },
  {
    keys: 'Esc',
    description: 'Cerrar modales',
    category: 'Navegación',
  },
  {
    keys: 'Ctrl + K',
    description: 'Abrir búsqueda',
    category: 'Búsqueda',
  },
  {
    keys: 'Ctrl + /',
    description: 'Mostrar atajos',
    category: 'Ayuda',
  },
  {
    keys: 'Ctrl + D',
    description: 'Ir a Dashboard',
    category: 'Navegación',
  },
  {
    keys: 'Ctrl + G',
    description: 'Ir a Generar',
    category: 'Navegación',
  },
  {
    keys: 'Ctrl + T',
    description: 'Ir a Tareas',
    category: 'Navegación',
  },
  {
    keys: 'Ctrl + F',
    description: 'Ir a Favoritos',
    category: 'Navegación',
  },
];

export default function KeyboardShortcuts() {
  const [isOpen, setIsOpen] = useState(false);

  useHotkeys('ctrl+/,cmd+/', () => setIsOpen(true), { preventDefault: true });

  useEffect(() => {
    const handleOpenShortcuts = () => setIsOpen(true);
    window.addEventListener('bul_open_shortcuts', handleOpenShortcuts);
    return () => window.removeEventListener('bul_open_shortcuts', handleOpenShortcuts);
  }, []);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-50"
            onClick={() => setIsOpen(false)}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 flex items-center justify-center z-50 p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
              <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FiCommand size={24} className="text-primary-600" />
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                    Atajos de Teclado
                  </h3>
                </div>
                <button onClick={() => setIsOpen(false)} className="btn-icon">
                  <FiX size={20} />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-6">
                <div className="space-y-6">
                  {['Generación', 'Navegación', 'Búsqueda', 'Ayuda'].map((category) => {
                    const categoryShortcuts = shortcuts.filter((s) => s.category === category);
                    if (categoryShortcuts.length === 0) return null;

                    return (
                      <div key={category}>
                        <h4 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-3">
                          {category}
                        </h4>
                        <div className="space-y-2">
                          {categoryShortcuts.map((shortcut, index) => (
                            <div
                              key={index}
                              className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                            >
                              <span className="text-sm text-gray-700 dark:text-gray-300">
                                {shortcut.description}
                              </span>
                              <kbd className="px-3 py-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded text-xs font-mono">
                                {shortcut.keys}
                              </kbd>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="p-6 border-t border-gray-200 dark:border-gray-700">
                <button
                  onClick={() => setIsOpen(false)}
                  className="btn btn-primary w-full"
                >
                  Cerrar
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

