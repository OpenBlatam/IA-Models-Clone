'use client';

import { useState, useEffect } from 'react';
import { Command } from 'cmdk';
import { Search, X, Command as CommandIcon, LayoutDashboard } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface SearchResult {
  id: string;
  type: 'command' | 'tab' | 'feature';
  title: string;
  description: string;
  action: () => void;
}

interface SearchBarProps {
  onResultSelect: (result: SearchResult) => void;
  tabs?: Array<{ id: string; label: string; action: () => void }>;
}

export default function SearchBar({ onResultSelect, tabs = [] }: SearchBarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
        setQuery('');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  const commands: SearchResult[] = [
    ...tabs.map((tab) => ({
      id: `tab-${tab.id}`,
      type: 'tab' as const,
      title: tab.label,
      description: `Ir a ${tab.label}`,
      action: tab.action,
    })),
    {
      id: 'cmd-move',
      type: 'command' as const,
      title: 'Mover a posición',
      description: 'move to (x, y, z)',
      action: () => {
        onResultSelect({
          id: 'cmd-move',
          type: 'command',
          title: 'Mover a posición',
          description: 'move to (x, y, z)',
          action: () => {},
        });
      },
    },
    {
      id: 'cmd-home',
      type: 'command' as const,
      title: 'Ir a home',
      description: 'go home',
      action: () => {
        onResultSelect({
          id: 'cmd-home',
          type: 'command',
          title: 'Ir a home',
          description: 'go home',
          action: () => {},
        });
      },
    },
    {
      id: 'cmd-stop',
      type: 'command' as const,
      title: 'Detener',
      description: 'stop',
      action: () => {
        onResultSelect({
          id: 'cmd-stop',
          type: 'command',
          title: 'Detener',
          description: 'stop',
          action: () => {},
        });
      },
    },
    {
      id: 'cmd-status',
      type: 'command' as const,
      title: 'Estado',
      description: 'status',
      action: () => {
        onResultSelect({
          id: 'cmd-status',
          type: 'command',
          title: 'Estado',
          description: 'status',
          action: () => {},
        });
      },
    },
  ];

  const filteredCommands = commands.filter((cmd) => {
    if (!query.trim()) return false;
    const searchTerm = query.toLowerCase();
    return (
      cmd.title.toLowerCase().includes(searchTerm) ||
      cmd.description.toLowerCase().includes(searchTerm)
    );
  });

  if (!isOpen) {
    return (
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={() => setIsOpen(true)}
        className="flex items-center gap-tesla-sm px-tesla-md py-tesla-sm bg-white border border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all text-sm font-medium"
      >
        <Search className="w-4 h-4 text-tesla-gray-dark" />
        <span className="hidden sm:inline">Buscar...</span>
        <kbd className="hidden sm:inline px-tesla-sm py-tesla-xs bg-gray-100 border border-gray-300 rounded text-xs text-tesla-gray-dark">Ctrl+K</kbd>
      </motion.button>
    );
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/30 backdrop-blur-sm"
            onClick={() => {
              setIsOpen(false);
              setQuery('');
            }}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-tesla-md"
            onClick={(e) => e.stopPropagation()}
          >
            <Command className="w-full max-w-2xl bg-white rounded-lg border border-gray-200 shadow-tesla-lg overflow-hidden">
              <div className="flex items-center gap-tesla-sm p-tesla-md border-b border-gray-200">
                <Search className="w-5 h-5 text-tesla-gray-dark" />
                <Command.Input
                  value={query}
                  onValueChange={setQuery}
                  placeholder="Buscar comandos, tabs, características..."
                  className="flex-1 bg-transparent text-tesla-black placeholder-tesla-gray-light focus:outline-none text-base"
                  autoFocus
                />
                <button
                  onClick={() => {
                    setIsOpen(false);
                    setQuery('');
                  }}
                  className="p-tesla-xs hover:bg-gray-100 rounded transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
                  aria-label="Cerrar búsqueda"
                >
                  <X className="w-4 h-4 text-tesla-gray-dark" />
                </button>
              </div>

              <Command.List className="max-h-96 overflow-y-auto p-tesla-sm scrollbar-hide">
                <Command.Empty className="p-tesla-xl text-center text-tesla-gray-dark">
                  No se encontraron resultados para "{query}"
                </Command.Empty>

                {!query && (
                  <div className="p-tesla-md text-sm text-tesla-gray-dark border-b border-gray-200">
                    <p className="mb-tesla-sm font-medium">Sugerencias:</p>
                    <div className="flex flex-wrap gap-tesla-sm items-center">
                      <kbd className="px-tesla-sm py-tesla-xs bg-gray-100 border border-gray-300 rounded text-xs text-tesla-black">Ctrl+K</kbd>
                      <span className="text-tesla-gray-dark">para buscar</span>
                      <kbd className="px-tesla-sm py-tesla-xs bg-gray-100 border border-gray-300 rounded text-xs text-tesla-black">Esc</kbd>
                      <span className="text-tesla-gray-dark">para cerrar</span>
                    </div>
                  </div>
                )}

                {filteredCommands.length > 0 && (
                  <Command.Group heading="Resultados">
                    {filteredCommands.map((result) => (
                      <Command.Item
                        key={result.id}
                        onSelect={() => {
                          result.action();
                          setIsOpen(false);
                          setQuery('');
                        }}
                        className="flex items-start gap-tesla-sm p-tesla-sm rounded-md cursor-pointer hover:bg-gray-50 transition-colors aria-selected:bg-tesla-blue/10 aria-selected:text-tesla-blue min-h-[44px]"
                      >
                        {result.type === 'command' ? (
                          <CommandIcon className="w-5 h-5 text-tesla-blue mt-0.5 flex-shrink-0" />
                        ) : (
                          <LayoutDashboard className="w-5 h-5 text-tesla-blue mt-0.5 flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-tesla-black font-medium truncate">{result.title}</p>
                          <p className="text-sm text-tesla-gray-dark truncate">{result.description}</p>
                        </div>
                        <span className="text-xs text-tesla-gray-light uppercase font-medium flex-shrink-0">
                          {result.type}
                        </span>
                      </Command.Item>
                    ))}
                  </Command.Group>
                )}
              </Command.List>
            </Command>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
