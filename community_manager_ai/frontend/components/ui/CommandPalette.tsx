'use client';

import { useState, useEffect, useRef } from 'react';
import { Search, Command } from 'lucide-react';
import * as Dialog from '@radix-ui/react-dialog';
import { cn } from '@/lib/utils';
import { useRouter as useI18nRouter } from '@/i18n/routing';
import { motion, AnimatePresence } from 'framer-motion';

interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon?: React.ReactNode;
  action: () => void;
  category?: string;
}

interface CommandPaletteProps {
  commands?: CommandItem[];
  trigger?: React.ReactNode;
}

const defaultCommands: CommandItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    description: 'Ir al dashboard principal',
    action: () => {},
    category: 'Navegación',
  },
  {
    id: 'posts',
    label: 'Posts',
    description: 'Gestionar posts',
    action: () => {},
    category: 'Navegación',
  },
  {
    id: 'memes',
    label: 'Memes',
    description: 'Gestionar memes',
    action: () => {},
    category: 'Navegación',
  },
  {
    id: 'calendar',
    label: 'Calendario',
    description: 'Ver calendario',
    action: () => {},
    category: 'Navegación',
  },
];

export const CommandPalette = ({ commands = defaultCommands, trigger }: CommandPaletteProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useI18nRouter();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const filteredCommands = commands.filter((cmd) =>
    cmd.label.toLowerCase().includes(search.toLowerCase()) ||
    cmd.description?.toLowerCase().includes(search.toLowerCase())
  );

  const handleSelect = (command: CommandItem) => {
    command.action();
    setIsOpen(false);
    setSearch('');
    setSelectedIndex(0);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % filteredCommands.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + filteredCommands.length) % filteredCommands.length);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filteredCommands[selectedIndex]) {
        handleSelect(filteredCommands[selectedIndex]);
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  return (
    <>
      {trigger ? (
        <div onClick={() => setIsOpen(true)}>{trigger}</div>
      ) : (
        <button
          type="button"
          onClick={() => setIsOpen(true)}
          className="flex items-center gap-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          aria-label="Abrir paleta de comandos"
        >
          <Search className="h-4 w-4" />
          <span className="hidden sm:inline">Buscar...</span>
          <kbd className="hidden sm:inline-flex h-5 select-none items-center gap-1 rounded border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 px-1.5 font-mono text-[10px] font-medium text-gray-500 dark:text-gray-400">
            <Command className="h-3 w-3" />K
          </kbd>
        </button>
      )}

      <Dialog.Root open={isOpen} onOpenChange={setIsOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm" />
          <Dialog.Content
            className={cn(
              'fixed left-1/2 top-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2',
              'rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-xl'
            )}
            onKeyDown={handleKeyDown}
          >
            <div className="p-2">
              <div className="flex items-center gap-2 border-b border-gray-200 dark:border-gray-700 px-3 pb-2">
                <Search className="h-4 w-4 text-gray-400 dark:text-gray-500" />
                <input
                  ref={inputRef}
                  type="text"
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value);
                    setSelectedIndex(0);
                  }}
                  placeholder="Buscar comandos..."
                  className="flex-1 bg-transparent py-2 text-sm outline-none text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500"
                />
              </div>

              <div className="max-h-96 overflow-y-auto py-2">
                <AnimatePresence>
                  {filteredCommands.length === 0 ? (
                    <div className="px-4 py-8 text-center text-sm text-gray-500 dark:text-gray-400">
                      No se encontraron comandos
                    </div>
                  ) : (
                    filteredCommands.map((command, index) => (
                      <motion.div
                        key={command.id}
                        initial={{ opacity: 0, y: -5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        className={cn(
                          'flex cursor-pointer items-center gap-3 px-3 py-2 transition-colors',
                          index === selectedIndex
                            ? 'bg-primary-50 dark:bg-primary-900/20'
                            : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                        )}
                        onClick={() => handleSelect(command)}
                      >
                        {command.icon}
                        <div className="flex-1">
                          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            {command.label}
                          </p>
                          {command.description && (
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                              {command.description}
                            </p>
                          )}
                        </div>
                      </motion.div>
                    ))
                  )}
                </AnimatePresence>
              </div>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </>
  );
};

