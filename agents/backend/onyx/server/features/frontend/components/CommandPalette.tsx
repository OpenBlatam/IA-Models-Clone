'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiCommand, FiX, FiSearch, FiFileText, FiSettings, FiUser } from 'react-icons/fi';
import { useHotkeys } from 'react-hotkeys-hook';
import { useAppStore } from '@/store/app-store';

interface Command {
  id: string;
  label: string;
  icon: React.ComponentType<{ size?: number }>;
  action: () => void;
  category: string;
}

export default function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const { setActiveView } = useAppStore();

  useHotkeys('ctrl+k,cmd+k', () => {
    setIsOpen(true);
  }, { preventDefault: true });

  const commands: Command[] = [
    {
      id: 'dashboard',
      label: 'Ir a Dashboard',
      icon: FiFileText,
      action: () => {
        setActiveView('dashboard');
        setIsOpen(false);
      },
      category: 'Navegación',
    },
    {
      id: 'generate',
      label: 'Generar Documento',
      icon: FiFileText,
      action: () => {
        setActiveView('generate');
        setIsOpen(false);
      },
      category: 'Navegación',
    },
    {
      id: 'tasks',
      label: 'Ver Tareas',
      icon: FiFileText,
      action: () => {
        setActiveView('tasks');
        setIsOpen(false);
      },
      category: 'Navegación',
    },
    {
      id: 'settings',
      label: 'Configuración',
      icon: FiSettings,
      action: () => {
        // Open settings panel
        setIsOpen(false);
      },
      category: 'Sistema',
    },
  ];

  const filteredCommands = query
    ? commands.filter((cmd) =>
        cmd.label.toLowerCase().includes(query.toLowerCase()) ||
        cmd.category.toLowerCase().includes(query.toLowerCase())
      )
    : commands;

  useEffect(() => {
    if (isOpen) {
      setQuery('');
    }
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-50"
            onClick={() => setIsOpen(false)}
          />
          <motion.div
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            className="fixed top-20 left-1/2 transform -translate-x-1/2 w-full max-w-2xl z-50 p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl overflow-hidden">
              <div className="flex items-center gap-3 p-4 border-b border-gray-200 dark:border-gray-700">
                <FiCommand size={20} className="text-gray-400" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Buscar comandos..."
                  className="flex-1 bg-transparent border-none outline-none text-gray-900 dark:text-white placeholder-gray-400"
                  autoFocus
                />
                <button
                  onClick={() => setIsOpen(false)}
                  className="btn-icon"
                >
                  <FiX size={18} />
                </button>
              </div>

              <div className="max-h-96 overflow-y-auto">
                {filteredCommands.length === 0 ? (
                  <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                    <p>No se encontraron comandos</p>
                  </div>
                ) : (
                  <div className="divide-y divide-gray-200 dark:divide-gray-700">
                    {['Navegación', 'Sistema'].map((category) => {
                      const categoryCommands = filteredCommands.filter(
                        (cmd) => cmd.category === category
                      );
                      if (categoryCommands.length === 0) return null;

                      return (
                        <div key={category}>
                          <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase bg-gray-50 dark:bg-gray-900">
                            {category}
                          </div>
                          {categoryCommands.map((command) => (
                            <button
                              key={command.id}
                              onClick={command.action}
                              className="w-full text-left px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-3"
                            >
                              <command.icon size={18} className="text-gray-400" />
                              <span className="text-sm text-gray-900 dark:text-white">
                                {command.label}
                              </span>
                            </button>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                  <div className="flex items-center gap-4">
                    <span>↑↓ Navegar</span>
                    <span>Enter Seleccionar</span>
                    <span>Esc Cerrar</span>
                  </div>
                  <kbd className="px-2 py-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded">
                    Ctrl+K
                  </kbd>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}


