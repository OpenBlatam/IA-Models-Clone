'use client';

import { useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Keyboard } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './Dialog';
import { Badge } from './Badge';

interface Shortcut {
  keys: string[];
  description: string;
  category?: string;
}

interface KeyboardShortcutsProps {
  shortcuts: Shortcut[];
  showTrigger?: boolean;
}

export default function KeyboardShortcuts({ shortcuts, showTrigger = true }: KeyboardShortcutsProps) {
  const [isOpen, setIsOpen] = useState(false);

  // Close on Escape
  useHotkeys('escape', () => setIsOpen(false), { enabled: isOpen });

  const groupedShortcuts = shortcuts.reduce((acc, shortcut) => {
    const category = shortcut.category || 'General';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(shortcut);
    return acc;
  }, {} as Record<string, Shortcut[]>);

  return (
    <>
      {showTrigger && (
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
          <DialogTrigger asChild>
            <button
              className="fixed bottom-4 right-4 p-3 bg-tesla-blue text-white rounded-full shadow-tesla-lg hover:shadow-tesla-xl transition-all z-50 min-h-[56px] min-w-[56px] flex items-center justify-center"
              aria-label="Ver atajos de teclado"
            >
              <Keyboard className="w-5 h-5" />
            </button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Keyboard className="w-5 h-5 text-tesla-blue" />
                Atajos de Teclado
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-6 mt-4">
              {Object.entries(groupedShortcuts).map(([category, categoryShortcuts]) => (
                <div key={category}>
                  <h3 className="text-sm font-semibold text-tesla-gray-dark mb-3 uppercase tracking-wide">
                    {category}
                  </h3>
                  <div className="space-y-2">
                    {categoryShortcuts.map((shortcut, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-3 bg-gray-50 rounded-md hover:bg-gray-100 transition-colors"
                      >
                        <span className="text-sm text-tesla-black">{shortcut.description}</span>
                        <div className="flex items-center gap-1">
                          {shortcut.keys.map((key, keyIndex) => (
                            <Badge key={keyIndex} variant="default" size="sm" className="font-mono">
                              {key}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </DialogContent>
        </Dialog>
      )}
    </>
  );
}

