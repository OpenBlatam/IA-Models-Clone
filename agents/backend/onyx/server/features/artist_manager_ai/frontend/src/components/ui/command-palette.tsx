'use client';

import { useState, useEffect, useCallback } from 'react';
import { Search, Command } from 'lucide-react';
import { Dialog } from '@headlessui/react';
import { cn } from '@/lib/utils';
import { Input } from '@/components/ui/input';

interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon?: React.ReactNode;
  action: () => void;
  keywords?: string[];
}

interface CommandPaletteProps {
  items: CommandItem[];
  isOpen: boolean;
  onClose: () => void;
  placeholder?: string;
}

const CommandPalette = ({
  items,
  isOpen,
  onClose,
  placeholder = 'Buscar comandos...',
}: CommandPaletteProps) => {
  const [search, setSearch] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);

  const filteredItems = items.filter((item) => {
    const searchLower = search.toLowerCase();
    return (
      item.label.toLowerCase().includes(searchLower) ||
      item.description?.toLowerCase().includes(searchLower) ||
      item.keywords?.some((keyword) => keyword.toLowerCase().includes(searchLower))
    );
  });

  const handleSelect = useCallback(
    (item: CommandItem) => {
      item.action();
      setSearch('');
      setSelectedIndex(0);
      onClose();
    },
    [onClose]
  );

  useEffect(() => {
    if (!isOpen) {
      setSearch('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, filteredItems.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter' && filteredItems[selectedIndex]) {
        e.preventDefault();
        handleSelect(filteredItems[selectedIndex]);
      } else if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredItems, selectedIndex, handleSelect, onClose]);

  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-50">
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />
      <div className="fixed inset-0 flex items-start justify-center p-4 pt-[20vh]">
        <Dialog.Panel className="w-full max-w-2xl bg-white rounded-lg shadow-xl">
          <div className="flex items-center border-b px-4">
            <Search className="w-5 h-5 text-gray-400 mr-3" />
            <Input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={placeholder}
              className="border-0 focus:ring-0 text-lg py-4"
              autoFocus
            />
          </div>
          <div className="max-h-96 overflow-y-auto p-2">
            {filteredItems.length === 0 ? (
              <div className="px-4 py-8 text-center text-gray-500">
                No se encontraron resultados
              </div>
            ) : (
              filteredItems.map((item, index) => (
                <button
                  key={item.id}
                  onClick={() => handleSelect(item)}
                  className={cn(
                    'w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors',
                    index === selectedIndex
                      ? 'bg-blue-50 text-blue-900'
                      : 'hover:bg-gray-50 text-gray-900'
                  )}
                >
                  {item.icon && <div className="flex-shrink-0">{item.icon}</div>}
                  <div className="flex-1 min-w-0">
                    <div className="font-medium">{item.label}</div>
                    {item.description && (
                      <div className="text-sm text-gray-500 truncate">{item.description}</div>
                    )}
                  </div>
                </button>
              ))
            )}
          </div>
          <div className="flex items-center justify-between border-t px-4 py-2 text-xs text-gray-500">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <kbd className="px-2 py-1 bg-gray-100 rounded">↑↓</kbd>
                navegar
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-2 py-1 bg-gray-100 rounded">Enter</kbd>
                seleccionar
              </span>
            </div>
            <span className="flex items-center gap-1">
              <kbd className="px-2 py-1 bg-gray-100 rounded">Esc</kbd>
              cerrar
            </span>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
};

export { CommandPalette };

