/**
 * Command palette component for quick actions
 */

'use client';

import React, { useEffect, useState } from 'react';
import { cn } from '@/lib/utils/cn';
import { Search, Command, X } from 'lucide-react';
import { Button } from './Button';
import { Backdrop } from './Backdrop';
import { useRouter } from 'next/navigation';

export interface CommandAction {
  id: string;
  label: string;
  description?: string;
  icon?: React.ReactNode;
  action: () => void;
  keywords?: string[];
  category?: string;
}

export interface CommandPaletteProps {
  actions: CommandAction[];
  isOpen: boolean;
  onClose: () => void;
  trigger?: React.ReactNode;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  actions,
  isOpen,
  onClose,
  trigger,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const filteredActions = React.useMemo(() => {
    if (!searchTerm) {
      return actions;
    }

    const searchLower = searchTerm.toLowerCase();
    return actions.filter((action) => {
      const matchesLabel = action.label.toLowerCase().includes(searchLower);
      const matchesDescription = action.description?.toLowerCase().includes(searchLower);
      const matchesKeywords = action.keywords?.some((keyword) =>
        keyword.toLowerCase().includes(searchLower)
      );
      return matchesLabel || matchesDescription || matchesKeywords;
    });
  }, [actions, searchTerm]);

  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
      setSearchTerm('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault();
        if (isOpen) {
          onClose();
        } else {
          // This would be handled by parent component
        }
      }

      if (!isOpen) {
        return;
      }

      if (event.key === 'Escape') {
        onClose();
      } else if (event.key === 'ArrowDown') {
        event.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % filteredActions.length);
      } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        setSelectedIndex((prev) => (prev - 1 + filteredActions.length) % filteredActions.length);
      } else if (event.key === 'Enter') {
        event.preventDefault();
        if (filteredActions[selectedIndex]) {
          filteredActions[selectedIndex].action();
          onClose();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, filteredActions, selectedIndex, onClose]);

  if (!isOpen) {
    return trigger || null;
  }

  const handleSelect = (action: CommandAction) => {
    action.action();
    onClose();
  };

  const groupedActions = React.useMemo(() => {
    const groups: Record<string, CommandAction[]> = {};
    filteredActions.forEach((action) => {
      const category = action.category || 'General';
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(action);
    });
    return groups;
  }, [filteredActions]);

  return (
    <>
      <Backdrop isOpen={isOpen} onClick={onClose} />
      <div
        className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]"
        role="dialog"
        aria-modal="true"
        aria-label="Paleta de comandos"
      >
        <div className="w-full max-w-2xl bg-background border rounded-lg shadow-xl">
          <div className="flex items-center gap-2 p-3 border-b">
            <Search className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
            <input
              ref={inputRef}
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Buscar comandos..."
              className="flex-1 bg-transparent border-none outline-none"
              aria-label="Buscar comandos"
            />
            <kbd className="hidden sm:inline-flex items-center gap-1 px-2 py-1 text-xs font-semibold text-muted-foreground bg-muted rounded border">
              <Command className="h-3 w-3" aria-hidden="true" />K
            </kbd>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              aria-label="Cerrar paleta de comandos"
              tabIndex={0}
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </Button>
          </div>

          <div className="max-h-96 overflow-y-auto p-2">
            {filteredActions.length === 0 ? (
              <div className="py-8 text-center text-muted-foreground">
                No se encontraron comandos
              </div>
            ) : (
              Object.entries(groupedActions).map(([category, categoryActions]) => (
                <div key={category}>
                  {category !== 'General' && (
                    <div className="px-2 py-1 text-xs font-semibold text-muted-foreground uppercase">
                      {category}
                    </div>
                  )}
                  {categoryActions.map((action, index) => {
                    const globalIndex = filteredActions.indexOf(action);
                    const isSelected = globalIndex === selectedIndex;

                    return (
                      <div
                        key={action.id}
                        className={cn(
                          'flex items-center gap-3 px-3 py-2 rounded cursor-pointer transition-colors',
                          isSelected ? 'bg-accent' : 'hover:bg-accent'
                        )}
                        onClick={() => handleSelect(action)}
                        onMouseEnter={() => setSelectedIndex(globalIndex)}
                        role="option"
                        aria-selected={isSelected}
                        tabIndex={0}
                      >
                        {action.icon && (
                          <div className="text-muted-foreground" aria-hidden="true">
                            {action.icon}
                          </div>
                        )}
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium">{action.label}</div>
                          {action.description && (
                            <div className="text-xs text-muted-foreground mt-0.5">
                              {action.description}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </>
  );
};



