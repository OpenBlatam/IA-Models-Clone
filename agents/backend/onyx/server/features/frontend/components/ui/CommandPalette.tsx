'use client';

import { ReactNode, useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiSearch, FiX, FiCommand } from 'react-icons/fi';
import { useKeyPress } from '@/hooks';
import { cn } from '@/utils/classNames';

interface Command {
  id: string;
  label: string;
  description?: string;
  icon?: ReactNode;
  keywords?: string[];
  action: () => void;
  group?: string;
}

interface CommandPaletteProps {
  commands: Command[];
  isOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  placeholder?: string;
}

export function CommandPalette({
  commands,
  isOpen: controlledOpen,
  onOpenChange,
  placeholder = 'Buscar comandos...',
}: CommandPaletteProps) {
  const [internalOpen, setInternalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const isControlled = controlledOpen !== undefined;
  const isOpen = isControlled ? controlledOpen : internalOpen;

  // Open with Cmd+K or Ctrl+K
  const cmdK = useKeyPress(['Meta', 'k']);
  const ctrlK = useKeyPress(['Control', 'k']);

  useEffect(() => {
    if (cmdK || ctrlK) {
      if (!isControlled) {
        setInternalOpen(true);
      }
      onOpenChange?.(true);
    }
  }, [cmdK, ctrlK, isControlled, onOpenChange]);

  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
    } else {
      setSearchQuery('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  const filteredCommands = commands.filter((cmd) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      cmd.label.toLowerCase().includes(query) ||
      cmd.description?.toLowerCase().includes(query) ||
      cmd.keywords?.some((k) => k.toLowerCase().includes(query))
    );
  });

  const groupedCommands = filteredCommands.reduce((acc, cmd) => {
    const group = cmd.group || 'Otros';
    if (!acc[group]) acc[group] = [];
    acc[group].push(cmd);
    return acc;
  }, {} as Record<string, Command[]>);

  const handleSelect = (command: Command) => {
    command.action();
    if (!isControlled) {
      setInternalOpen(false);
    }
    onOpenChange?.(false);
  };

  const handleClose = () => {
    if (!isControlled) {
      setInternalOpen(false);
    }
    onOpenChange?.(false);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'Escape') {
        handleClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, filteredCommands.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          handleSelect(filteredCommands[selectedIndex]);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredCommands, selectedIndex]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50"
        onClick={handleClose}
      />
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: -20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: -20 }}
        className="relative w-full max-w-2xl mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl overflow-hidden">
          <div className="flex items-center gap-3 px-4 border-b border-gray-200 dark:border-gray-700">
            <FiSearch className="text-gray-400" size={20} />
            <input
              ref={inputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setSelectedIndex(0);
              }}
              placeholder={placeholder}
              className="flex-1 py-3 bg-transparent border-0 focus:outline-none text-gray-900 dark:text-white placeholder-gray-400"
            />
            <button
              onClick={handleClose}
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
            >
              <FiX size={18} />
            </button>
          </div>

          <div className="max-h-96 overflow-y-auto">
            {filteredCommands.length === 0 ? (
              <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                No se encontraron comandos
              </div>
            ) : (
              Object.entries(groupedCommands).map(([group, groupCommands]) => (
                <div key={group}>
                  {group !== 'Otros' && (
                    <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase">
                      {group}
                    </div>
                  )}
                  {groupCommands.map((cmd, index) => {
                    const globalIndex = filteredCommands.indexOf(cmd);
                    const isSelected = globalIndex === selectedIndex;

                    return (
                      <button
                        key={cmd.id}
                        onClick={() => handleSelect(cmd)}
                        className={cn(
                          'w-full flex items-center gap-3 px-4 py-3 text-left',
                          'hover:bg-gray-100 dark:hover:bg-gray-700',
                          'transition-colors',
                          isSelected && 'bg-gray-100 dark:bg-gray-700'
                        )}
                      >
                        {cmd.icon && <span className="flex-shrink-0">{cmd.icon}</span>}
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-gray-900 dark:text-white">
                            {cmd.label}
                          </div>
                          {cmd.description && (
                            <div className="text-sm text-gray-500 dark:text-gray-400">
                              {cmd.description}
                            </div>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              ))
            )}
          </div>

          <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">↑</kbd>
                <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">↓</kbd>
                Navegar
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">Enter</kbd>
                Seleccionar
              </span>
            </div>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">Esc</kbd>
              Cerrar
            </span>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

